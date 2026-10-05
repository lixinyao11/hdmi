"""AnyBody Stage 3 on HDMI: PPO in the latent space of the frozen Stage 1 decoder.

Port of AnyBody's ``LatentRLActorCritic`` (residual adapter) + ``LatentPPO``.

    mu      = E2(keypoints, box goal, proprio)          # Stage 2 encoder, frozen
    delta   = g(mu, inputs)                             # residual adapter, trained, starts ~0
    z       ~ N(normalize(mu + delta), sigma)           # the PPO action (16-d)
    action  = D(normalize(z), proprio)                  # Stage 1 decoder, frozen

At delta = 0 and the mean latent this reproduces the Stage 2 policy exactly (safe start).

Reward (AnyBody LatentRLRewardsCfg, plus the box): it only scores what the policy can see.
    3.0 * exp(-mean_visible |kp pos err|^2 / 0.3^2)    world frame, visible keypoints only
  + 3.0 * exp(-mean_visible |kp vel err|^2 / 1.0^2)
  + 1.0 * exp(-|anchor pos err|^2 / 0.3^2) + 0.5 * exp(-anchor_angle^2 / 0.4^2)   anti-drift
  + 1.0 * env object_tracking group  (only when the box goal is visible to the policy)
  + 1.0 * env loco group             (penalties + survival, unchanged)
  + 0.125 * env tracking group       (weak full-body reference: naturalness of unseen bodies)
Keypoint terms are 0 for episodes with no visible keypoint.

Differences from AnyBody, deliberate:
- The residual adapter is an MLP over [mu, masked inputs, mask flags] rather than a small
  transformer; masked inputs are zeroed and their flags given explicitly.
- The critic is asymmetric (sees the unmasked command and keypoints plus the mask flags); AnyBody's
  first milestone used a symmetric critic and listed this as the next enhancement.
- The keypoint mask distribution is held at Stage 2's final phase (8-mode mix, p_see 0.4, box goal
  hidden with p 0.2, never everything hidden).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tensordict import TensorDictBase

import active_adaptation as aa
from active_adaptation.learning.ppo.common import ACTION_KEY, CMD_KEY, DONE_KEY, OBS_KEY, TERM_KEY, GAE
from active_adaptation.learning.ppo.ppo_base import PPOBase
from active_adaptation.utils.wandb import parse_checkpoint

from .latent_kp_distill import KP_KEY, LatentKpDistillConfig, LatentKpDistillPolicy


RW_KEY = "rl_reward_"
Z_KEY, LOGP_KEY = "latent_z", "latent_logp"
KPM_KEY, OBJM_KEY = "rl_kp_mask", "rl_obj_mask"


@dataclass
class LatentPPOConfig:
    _target_: str = f"{__package__}.latent_ppo.LatentPPOConfig"
    name: str = "hdmi_latent_ppo"

    stage2_checkpoint: str | None = None  # hdmi_latent_kp_distill checkpoint (run: spec or path)
    train_every: int = 24
    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, KP_KEY, RW_KEY)

    # Policy (AnyBody: adapter=residual, init_latent_std=0.1).
    residual_hidden_dims: Tuple[int, ...] = (512, 256)
    residual_init_gain: float = 0.01
    init_latent_std: float = 0.1
    critic_hidden_dims: Tuple[int, ...] = (1024, 512, 256, 128)

    # PPO (AnyBody G1FlatMUSEKpLatentRLKp5RunnerCfg.algorithm).
    lr: float = 1e-4
    num_epochs: int = 5
    num_minibatches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    lr_min: float = 1e-5  # floor of the adaptive schedule (rsl_rl uses 1e-5)
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.0
    critic_warmup_iters: int = 100

    # Reward weights / scales.
    w_kp_pos: float = 3.0
    kp_pos_std: float = 0.3
    w_kp_vel: float = 3.0
    kp_vel_std: float = 1.0
    w_anchor_pos: float = 1.0
    anchor_pos_std: float = 0.3
    w_anchor_ori: float = 0.5
    anchor_ori_std: float = 0.4
    w_object: float = 1.0
    w_loco: float = 1.0
    w_ref: float = 0.125

    # Mask distribution (held at Stage 2's final phase) and eval settings.
    p_see: float = 0.4
    obj_hidden_p: float = 0.2
    eval_mode: str = "full"
    eval_object: str = "visible"
    # Eval-only extra keypoint modes (e.g. {"none": []}). Never set for training: Stage 2's final
    # phase samples uniformly over all modes, so this would change the training distribution.
    extra_modes: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        self.in_keys = tuple(self.in_keys)
        self.residual_hidden_dims = tuple(self.residual_hidden_dims)
        self.critic_hidden_dims = tuple(self.critic_hidden_dims)
        for key in (CMD_KEY, OBS_KEY, KP_KEY, RW_KEY):
            if key not in self.in_keys:
                raise ValueError(f"in_keys {self.in_keys} must contain {key!r}")

    def get_class(self):
        return LatentPPOPolicy


cs = ConfigStore.instance()
cs.store("hdmi_latent_ppo", node=LatentPPOConfig, group="algo")


def _mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int, act=nn.ELU) -> nn.Sequential:
    layers = []
    for h in hidden:
        layers += [nn.Linear(in_dim, h), act()]
        in_dim = h
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class _Rollout(nn.Module):
    def __init__(self, policy: "LatentPPOPolicy", mode: str):
        super().__init__()
        object.__setattr__(self, "policy", policy)
        self.mode = mode
        self.in_keys = [CMD_KEY, OBS_KEY, KP_KEY, "is_init"]
        self.out_keys = [ACTION_KEY] if mode == "deploy" else [ACTION_KEY, Z_KEY, LOGP_KEY, KPM_KEY, OBJM_KEY]

    @torch.no_grad()
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        p = self.policy
        kp_mask, _, obj_mask = p.s2.update_masks(tensordict, "train" if self.mode == "train" else "eval")
        feats = p.features(tensordict, kp_mask, obj_mask)
        mean = p.latent_mean(feats)
        if self.mode == "train":
            std = p.log_std.exp().expand_as(mean)
            z = mean + std * torch.randn_like(mean)
            logp = torch.distributions.Normal(mean, std).log_prob(z).sum(-1)
        else:
            z, logp = mean, torch.zeros(mean.shape[0], device=mean.device)
        action = p.s2.stage1.decode(F.normalize(z, dim=-1, eps=1e-8), feats["proprio"])
        tensordict.set(ACTION_KEY, action)
        if self.mode != "deploy":
            tensordict.set(Z_KEY, z)
            tensordict.set(LOGP_KEY, logp)
            tensordict.set(KPM_KEY, kp_mask.clone())
            tensordict.set(OBJM_KEY, obj_mask.clone())
        return tensordict


class LatentPPOPolicy(PPOBase):
    requires_rollout_value = False

    @classmethod
    def from_env(cls, cfg, env, device):
        return cls(cfg, env.observation_spec, env.action_spec, env.reward_spec, device, getattr(env, "base_env", env))

    def __init__(self, cfg: LatentPPOConfig, observation_spec, action_spec, reward_spec, device, env):
        super().__init__()
        self.cfg = cfg if isinstance(cfg, LatentPPOConfig) else LatentPPOConfig(**dict(cfg))
        if self.cfg.stage2_checkpoint is None:
            raise ValueError("algo.stage2_checkpoint is required")
        self.device = device
        object.__setattr__(self, "env", env)

        # --- frozen Stage 2 (+ its frozen Stage 1 decoder and teacher normalizers) ---
        ckpt = parse_checkpoint(self.cfg.stage2_checkpoint)
        ckpt.update()
        path = ckpt.get_path()
        print(f"[latent_ppo] stage2 checkpoint: {self.cfg.stage2_checkpoint} -> {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        s2_algo = OmegaConf.to_container(OmegaConf.create(state["cfg"]).algo, resolve=True)
        s2_algo.update(
            phase1_end_iter=0, phase2_end_iter=0, p_see_final=self.cfg.p_see,
            obj_hidden_p_final=self.cfg.obj_hidden_p, eval_mode=self.cfg.eval_mode,
            eval_object=self.cfg.eval_object, rollout_actor="student",
            in_keys=[CMD_KEY, OBS_KEY, KP_KEY],
        )
        if self.cfg.extra_modes:
            s2_algo["modes"] = {**s2_algo["modes"], **{k: list(v) for k, v in dict(self.cfg.extra_modes).items()}}
        s2_algo = {k: v for k, v in s2_algo.items() if k in LatentKpDistillConfig.__dataclass_fields__}
        s2 = LatentKpDistillPolicy(LatentKpDistillConfig(**s2_algo), observation_spec, action_spec, reward_spec, device, env)
        s2.load_state_dict(state["policy"])
        if hasattr(s2.kp_encoder, "module"):
            s2.kp_encoder = s2.kp_encoder.module  # frozen: no DDP
        s2.requires_grad_(False).eval()
        object.__setattr__(self, "s2", s2)  # not a submodule: never saved, never optimized
        print(f"[latent_ppo] Stage 2 loaded; keypoint bodies {s2.body_names}")

        # --- reward-only observation layout ---
        rw_group = env.observation_groups[RW_KEY]
        self.rw_split = dict(zip(rw_group.keys(), rw_group.split))
        for term in ("kp_pos_err", "kp_vel_err", "anchor_err"):
            if term not in self.rw_split:
                raise ValueError(f"{RW_KEY} group needs term {term!r}, has {list(self.rw_split)}")
        n_b = len(s2.body_names)
        for term in ("kp_pos_err", "kp_vel_err"):
            t = rw_group[term]
            names = [t.command_manager.tracking_body_names[i] for i in t.body_indices_tracking.tolist()]
            if names != s2.body_names:
                raise ValueError(f"{RW_KEY}.{term} bodies {names} != keypoint bodies {s2.body_names}")
        if self.rw_split["kp_vel_err"] != n_b * 3:
            raise ValueError("kp_vel_err must use exactly one future step (future_steps: [0])")

        # --- trainable parts ---
        latent_dim = s2.stage1.encoder.mu_head.out_features
        proprio_dim = observation_spec[OBS_KEY].shape[-1]
        kp_dim = observation_spec[KP_KEY].shape[-1]
        obj_dim = s2.obj_slice.stop - s2.obj_slice.start
        goal_dim = observation_spec[CMD_KEY].shape[-1]
        res_in = latent_dim + proprio_dim + kp_dim + n_b + obj_dim + 1
        self.residual = _mlp(res_in, self.cfg.residual_hidden_dims, latent_dim).to(device)
        last = self.residual[-1]
        nn.init.normal_(last.weight, std=self.cfg.residual_init_gain / math.sqrt(last.in_features))
        nn.init.zeros_(last.bias)
        self.log_std = nn.Parameter(torch.full((latent_dim,), math.log(self.cfg.init_latent_std), device=device))
        critic_in = goal_dim + proprio_dim + kp_dim + n_b + 1
        self.critic = _mlp(critic_in, self.cfg.critic_hidden_dims, 1).to(device)

        self.opt = torch.optim.Adam(list(self.residual.parameters()) + [self.log_std] + list(self.critic.parameters()), lr=self.cfg.lr)
        self.lr = self.cfg.lr
        self.gae = GAE(self.cfg.gamma, self.cfg.lam).to(device)
        if aa.is_distributed():
            for prm in self._trainable():
                dist.broadcast(prm.data, src=0)

    # ---------------------------------------------------------------- helpers
    def _trainable(self):
        return list(self.residual.parameters()) + [self.log_std] + list(self.critic.parameters())

    def features(self, tensordict, kp_mask: torch.Tensor, obj_mask: torch.Tensor, prefix=()) -> dict:
        """Normalized inputs (optionally from ("next", ...)), the frozen encoder mean, and masks."""
        key = (lambda k: (*prefix, k))
        td = {k: tensordict[key(k)] for k in (CMD_KEY, OBS_KEY, KP_KEY)}
        x = self.s2.prepare_inputs(td)
        obj = x["obj"]
        if self.cfg.eval_object == "on_target" and not self.training_mode:
            obj = self.s2.object_on_target_input(obj.shape[0])
        with torch.no_grad():
            mu = self.s2.unwrapped_encoder()(x["kp"], obj, x["proprio"], kp_mask, obj_mask)
        kp_flat = x["kp"].masked_fill(kp_mask.unsqueeze(-1), 0.0).flatten(-2)
        obj_in = x["obj"].masked_fill(obj_mask.unsqueeze(-1), 0.0)
        res_in = torch.cat([mu, x["proprio"], kp_flat, kp_mask.float(), obj_in, obj_mask.float().unsqueeze(-1)], -1)
        critic_in = torch.cat([x["goal"], x["proprio"], x["kp"].flatten(-2), kp_mask.float(), obj_mask.float().unsqueeze(-1)], -1)
        return {"proprio": x["proprio"], "mu": mu, "res_in": res_in, "critic_in": critic_in}

    training_mode = True

    def latent_mean(self, feats: dict) -> torch.Tensor:
        return F.normalize(feats["mu"] + self.residual(feats["res_in"]), dim=-1, eps=1e-8)

    def compute_reward(self, td: TensorDictBase, kp_mask: torch.Tensor, obj_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        c = self.cfg
        rw = td["next", RW_KEY]
        parts = dict(zip(self.rw_split, rw.split(list(self.rw_split.values()), dim=-1)))
        n_b = len(self.s2.body_names)
        visible = (~kp_mask).float()
        n_vis = visible.sum(-1)
        any_vis = (n_vis > 0).float()
        pos_sq = parts["kp_pos_err"].unflatten(-1, (n_b, 3)).square().sum(-1)
        vel_sq = parts["kp_vel_err"].unflatten(-1, (n_b, 3)).square().sum(-1)
        mean_vis = lambda v: (v * visible).sum(-1) / n_vis.clamp_min(1.0)
        r_kp_pos = c.w_kp_pos * torch.exp(-mean_vis(pos_sq) / c.kp_pos_std**2) * any_vis
        r_kp_vel = c.w_kp_vel * torch.exp(-mean_vis(vel_sq) / c.kp_vel_std**2) * any_vis
        anchor = parts["anchor_err"]
        r_anchor = c.w_anchor_pos * torch.exp(-anchor[..., :3].square().sum(-1) / c.anchor_pos_std**2) \
            + c.w_anchor_ori * torch.exp(-anchor[..., 3].square() / c.anchor_ori_std**2)
        # The env's reward groups are already multiplied by step_dt (IsaacLab convention, which
        # AnyBody's weights assume); scale the policy-side terms the same way.
        dt = float(self.env.step_dt)
        r_kp_pos, r_kp_vel, r_anchor = r_kp_pos * dt, r_kp_vel * dt, r_anchor * dt
        env_r = td["next", "reward"]
        r_obj = c.w_object * env_r["object_tracking"].squeeze(-1) * (~obj_mask).float()
        r_loco = c.w_loco * env_r["loco"].squeeze(-1)
        r_ref = c.w_ref * env_r["tracking"].squeeze(-1)
        total = r_kp_pos + r_kp_vel + r_anchor + r_obj + r_loco + r_ref
        with torch.no_grad():
            kp_err_cm = 100 * (mean_vis(pos_sq.sqrt()) * any_vis).sum() / any_vis.sum().clamp_min(1)
        info = {
            "reward/step_dt": dt,
            "reward/kp_pos": r_kp_pos.mean().item(), "reward/kp_vel": r_kp_vel.mean().item(),
            "reward/anchor": r_anchor.mean().item(), "reward/object": r_obj.mean().item(),
            "reward/loco": r_loco.mean().item(), "reward/ref": r_ref.mean().item(),
            "reward/total": total.mean().item(), "rl/visible_kp_pos_err_cm": kp_err_cm.item(),
        }
        return total, info

    def _sync_grads(self):
        if aa.is_distributed():
            for prm in self._trainable():
                if prm.grad is not None:
                    dist.all_reduce(prm.grad, op=dist.ReduceOp.AVG)

    # ------------------------------------------------------------- interface
    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        self.training_mode = mode == "train"
        return _Rollout(self, mode)

    def get_next_saved_keys(self):
        return (CMD_KEY, OBS_KEY, KP_KEY, RW_KEY)

    def train_op(self, td: TensorDictBase) -> dict:
        cfg = self.cfg
        N, T = td.shape
        kp_mask, obj_mask = td[KPM_KEY].bool(), td[OBJM_KEY].bool()
        reward, info = self.compute_reward(td, kp_mask, obj_mask)

        flat = lambda v: v.reshape(N * T, *v.shape[2:])
        tdf = td.reshape(N * T)  # the frozen encoder takes one batch dim
        kpm_f, objm_f = flat(kp_mask), flat(obj_mask)
        with torch.no_grad():
            feats = self.features(tdf, kpm_f, objm_f)
            nfeats = self.features(tdf, kpm_f, objm_f, prefix=("next",))
            value = self.critic(feats["critic_in"]).reshape(N, T, 1)
            next_value = self.critic(nfeats["critic_in"]).reshape(N, T, 1)
            terminated = td[TERM_KEY].reshape(N, T, 1)
            done = td[DONE_KEY].reshape(N, T, 1)
            adv, ret = self.gae(reward.unsqueeze(-1), terminated, done, value, next_value)
            adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)

        res_in, mu, critic_in = feats["res_in"], feats["mu"], feats["critic_in"]
        z, old_logp = flat(td[Z_KEY]), flat(td[LOGP_KEY])
        adv_f, ret_f, old_v = flat(adv).squeeze(-1), flat(ret).squeeze(-1), flat(value).squeeze(-1)

        with torch.no_grad():  # must be ~0: rollout and training see the same distribution
            mean0 = F.normalize(mu + self.residual(res_in), dim=-1, eps=1e-8)
            logp0 = torch.distributions.Normal(mean0, self.log_std.exp().expand_as(mean0)).log_prob(z).sum(-1)
            info["rl/initial_logratio_absmean"] = (logp0 - old_logp).abs().mean().item()

        warmup = self.num_updates < cfg.critic_warmup_iters
        stats = {k: 0.0 for k in ("pg", "vf", "kl", "clipfrac", "grad_norm")}
        n = 0
        mb = (N * T) // cfg.num_minibatches
        for _ in range(cfg.num_epochs):
            perm = torch.randperm(N * T, device=self.device)
            for i in range(cfg.num_minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                mean = F.normalize(mu[idx] + self.residual(res_in[idx]), dim=-1, eps=1e-8)
                std = self.log_std.exp().expand_as(mean)
                dist_now = torch.distributions.Normal(mean, std)
                logp = dist_now.log_prob(z[idx]).sum(-1)
                log_ratio = logp - old_logp[idx]
                ratio = log_ratio.exp()
                a = adv_f[idx]
                pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip_param, 1 + cfg.clip_param) * a).mean()
                v = self.critic(critic_in[idx]).squeeze(-1)
                v_clip = old_v[idx] + (v - old_v[idx]).clamp(-cfg.clip_param, cfg.clip_param)
                vf = torch.max((v - ret_f[idx]).square(), (v_clip - ret_f[idx]).square()).mean()
                ent = dist_now.entropy().sum(-1).mean()
                loss = cfg.value_loss_coef * vf
                if not warmup:
                    loss = loss + pg - cfg.entropy_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self._sync_grads()
                stats["grad_norm"] += nn.utils.clip_grad_norm_(self._trainable(), cfg.max_grad_norm).item()
                self.opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean()
                    stats["kl"] += kl.item()
                    stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip_param).float().mean().item()
                stats["pg"] += pg.item()
                stats["vf"] += vf.item()
                n += 1
                if not warmup and cfg.desired_kl > 0:  # rsl_rl "adaptive" schedule
                    if kl > 2.0 * cfg.desired_kl:
                        self.lr = max(cfg.lr_min, self.lr / 1.5)
                    elif kl < 0.5 * cfg.desired_kl:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr

        self.num_updates += 1
        with torch.no_grad():
            explained = 1 - (ret_f - old_v).var() / ret_f.var().clamp_min(1e-8)
            delta_norm = self.residual(res_in[: min(4096, len(res_in))]).norm(dim=-1).mean()
        info.update({f"rl/{k}": v / max(n, 1) for k, v in stats.items()})
        info.update({
            "rl/lr": self.lr, "rl/critic_warmup": float(warmup), "rl/explained_var": explained.item(),
            "rl/latent_std": self.log_std.exp().mean().item(), "rl/residual_norm": delta_norm.item(),
            "rl/value_mean": old_v.mean().item(), "rl/obj_hidden_frac": obj_mask.float().mean().item(),
            "rl/kp_visible_frac": (~kp_mask).float().mean().item(),
        })
        if aa.is_distributed():
            keys = sorted(info)
            vals = torch.tensor([info[k] for k in keys], device=self.device)
            dist.all_reduce(vals, op=dist.ReduceOp.AVG)
            info = dict(zip(keys, vals.tolist()))
        return info

    def state_dict(self):
        state = OrderedDict()
        state["residual"] = self.residual.state_dict()
        state["log_std"] = self.log_std.detach().clone()
        state["critic"] = self.critic.state_dict()
        state["opt"] = self.opt.state_dict()
        state["lr"] = self.lr
        state["num_updates"] = self.num_updates
        state["stage2_checkpoint"] = self.cfg.stage2_checkpoint
        state["last_iter"] = int(getattr(self.env, "current_iter", 0))
        return state

    def load_state_dict(self, state_dict, strict=True):
        if "residual" not in state_dict:
            raise KeyError("Checkpoint has no 'residual'; pass Stage 2 via algo.stage2_checkpoint.")
        self.residual.load_state_dict(state_dict["residual"], strict=strict)
        self.critic.load_state_dict(state_dict["critic"], strict=strict)
        with torch.no_grad():
            self.log_std.copy_(state_dict["log_std"])
        if "opt" in state_dict:
            self.opt.load_state_dict(state_dict["opt"])
        self.lr = state_dict.get("lr", self.lr)
        self.num_updates = state_dict.get("num_updates", 0)
        if hasattr(self.env, "set_progress"):
            self.env.set_progress(state_dict.get("last_iter", 0))
        return []
