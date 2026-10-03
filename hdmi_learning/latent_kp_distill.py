"""AnyBody Stage 2 on HDMI: sparse-keypoint encoder distilled in latent space.

Port of AnyBody's ``MuseKpLatentDistillation`` + ``LatentBottleneckMUSEKp``.

    teacher (frozen, Stage 1):  z_full = E1(full command, policy)     # sees everything
    student (trained):          z_kp   = E2(keypoints, object, policy)  # some keypoints hidden
    decoder (frozen, Stage 1):  a      = D(z, policy)                  # shared motor prior
    loss:  (1 - cos(z_kp, z_full)) + w_bc * ||D(z_kp) - D(z_full)||^2

Because D is frozen and shared, matching z_kp to z_full makes the student act like Stage 1.

Student tokens: [CLS] + one token per keypoint body (its future positions at the command's
future steps) + one object token (the command's object_spatial_error_local term, same as Stage 1)
+ the Stage 1 proprio tokens. Hidden keypoints are zeroed and removed from attention. The object
token is always visible: without it the robot has no goal for the carried object.

The transformer, [CLS], mu head, proprio projections and the object projection are warm-started
from the Stage 1 encoder; the keypoint projection and per-body embeddings are new.

Keypoint masks are sampled per env at episode start and held for the episode (AnyBody
``PartialMaskedMultiMotionCommand``). Curriculum (AnyBody ``MUSEKpLatentDemoCurriculumCfg``):
    [0, phase1_end)          bernoulli mode only, p_see = 1 (all keypoints visible)
    [phase1_end, phase2_end) bernoulli mode only, p_see ramps 1 -> p_see_final
    [phase2_end, ...)        uniform mix over all modes, bernoulli held at p_see_final

Differences from AnyBody, deliberate:
- Feet keypoints are the toe links (HDMI's obs body set has no ankles); torso_link is added to the
  obs body set by the exp config.
- Keypoint positions use the command's own future steps rather than AnyBody's 15-slot 0.5 s
  layout, so the student sees the same horizon as the Stage 1 teacher.
- Masking lives in the policy, so the env is unchanged apart from one extra observation group.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tensordict import TensorDictBase
from torch.nn.parallel import DistributedDataParallel as DDP

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.learning.ppo.common import ACTION_KEY, CMD_KEY, OBS_KEY
from active_adaptation.learning.ppo.ppo_base import PPOBase
from active_adaptation.utils.wandb import parse_checkpoint

from .latent_distill import FrozenTeacher, LatentStudent, TermTokenEncoder


KP_KEY = "command_kp"
KP_MASK_KEY = "kp_mask"  # [N, B] bool, True = hidden
KP_MODE_KEY = "kp_mode"  # [N] long, index into mode_names
Z_FULL_KEY = "z_full"
TEACHER_ACTION_KEY = "teacher_action"

TORSO = "torso_link"
L_WRIST, R_WRIST = "left_wrist_yaw_link", "right_wrist_yaw_link"
L_FOOT, R_FOOT = "left_toe_link", "right_toe_link"


def _default_modes() -> Dict[str, List[str]]:
    # AnyBody muse_kp5_latent_demo_mode_spec, ankles -> toes. Lists are the *visible* bodies.
    return {
        "full": [TORSO, L_WRIST, R_WRIST, L_FOOT, R_FOOT],
        "vr": [TORSO, L_WRIST, R_WRIST],
        "torso": [TORSO],
        "left_wrist": [L_WRIST],
        "right_wrist": [R_WRIST],
        "wrists": [L_WRIST, R_WRIST],
        "feet": [L_FOOT, R_FOOT],
        "bernoulli": [TORSO, L_WRIST, R_WRIST, L_FOOT, R_FOOT],
    }


@dataclass
class LatentKpDistillConfig:
    _target_: str = f"{__package__}.latent_kp_distill.LatentKpDistillConfig"
    name: str = "hdmi_latent_kp_distill"

    # Stage 1 (hdmi_latent_distill) checkpoint: run:<entity>/<project>/<id>:<iter> or a path.
    stage1_checkpoint: str | None = None

    train_every: int = 24
    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, KP_KEY)
    object_term: str = "object_spatial_error_local"

    modes: Dict[str, List[str]] = field(default_factory=_default_modes)
    bernoulli_mode: str = "bernoulli"

    # AnyBody MuseKpLatentDistillation.
    lr: float = 1e-3
    max_grad_norm: float = 1.0
    num_epochs: int = 5
    gradient_length: int = 15
    weight_latent: float = 1.0
    weight_behavior: float = 0.05

    # AnyBody MUSEKpLatentDemoCurriculumCfg (phase_until_learning_iterations=(1500, 6500, None)).
    phase1_end_iter: int = 1500
    phase2_end_iter: int = 6500
    p_see_final: float = 0.4

    # Eval/deploy: one fixed mode for every env ("bernoulli" uses eval_p_see).
    eval_mode: str = "full"
    eval_p_see: float = 0.4

    # "student" = DAgger (AnyBody). "stage1" drives with D(z_full): a diagnostic that should
    # reproduce the Stage 1 student's tracking.
    rollout_actor: str = "student"

    def __post_init__(self):
        self.in_keys = tuple(self.in_keys)
        self.modes = {str(k): [str(b) for b in v] for k, v in dict(self.modes).items()}
        for key in (CMD_KEY, OBS_KEY, KP_KEY):
            if key not in self.in_keys:
                raise ValueError(f"in_keys {self.in_keys} must contain {key!r}")
        if self.bernoulli_mode not in self.modes:
            raise ValueError(f"bernoulli_mode {self.bernoulli_mode!r} not in modes")
        if self.eval_mode not in self.modes:
            raise ValueError(f"eval_mode {self.eval_mode!r} not in modes {list(self.modes)}")
        if not 0 <= self.phase1_end_iter <= self.phase2_end_iter:
            raise ValueError("need 0 <= phase1_end_iter <= phase2_end_iter")
        if self.rollout_actor not in ("student", "stage1"):
            raise ValueError(f"rollout_actor must be 'student' or 'stage1', got {self.rollout_actor!r}")

    def get_class(self):
        return LatentKpDistillPolicy


cs = ConfigStore.instance()
cs.store("hdmi_latent_kp_distill", node=LatentKpDistillConfig, group="algo")


class KeypointEncoder(nn.Module):
    """[CLS] + keypoint tokens + object token + proprio tokens -> transformer -> unit-norm z."""

    def __init__(self, stage1: TermTokenEncoder, num_bodies: int, kp_token_dim: int, object_goal_index: int, d_model: int):
        super().__init__()
        self.num_bodies = num_bodies
        self.proprio_split = stage1.proprio_split
        self.latent_normalize = stage1.latent_normalize

        self.kp_proj = nn.Linear(kp_token_dim, d_model)
        self.body_emb = nn.Parameter(torch.randn(num_bodies, d_model) * 0.02)
        self.object_proj = nn.Linear(stage1.goal_split[object_goal_index], d_model)
        self.proprio_proj = nn.ModuleList(nn.Linear(s, d_model) for s in self.proprio_split)
        self.object_emb = nn.Parameter(torch.zeros(d_model))
        self.proprio_emb = nn.Parameter(torch.zeros(len(self.proprio_split), d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = stage1.transformer.layers[0]
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=layer.self_attn.num_heads,
                dim_feedforward=layer.linear1.out_features,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=len(stage1.transformer.layers),
            enable_nested_tensor=False,
        )
        self.mu_head = nn.Linear(d_model, stage1.mu_head.out_features)

        # Warm start everything that has a Stage 1 counterpart.
        with torch.no_grad():
            self.object_proj.load_state_dict(stage1.goal_proj[object_goal_index].state_dict())
            self.proprio_proj.load_state_dict(stage1.proprio_proj.state_dict())
            self.object_emb.copy_(stage1.token_emb[object_goal_index])
            self.proprio_emb.copy_(stage1.token_emb[len(stage1.goal_split):])
            self.cls_token.copy_(stage1.cls_token)
        self.transformer.load_state_dict(stage1.transformer.state_dict())
        self.mu_head.load_state_dict(stage1.mu_head.state_dict())

    def forward(self, kp: torch.Tensor, obj: torch.Tensor, proprio: torch.Tensor, kp_mask: torch.Tensor) -> torch.Tensor:
        """kp [N, B, S*3], obj [N, G_obj], proprio [N, P], kp_mask [N, B] (True = hidden) -> z."""
        n = kp.shape[0]
        kp_tokens = (self.kp_proj(kp) + self.body_emb).masked_fill(kp_mask.unsqueeze(-1), 0.0)
        obj_token = (self.object_proj(obj) + self.object_emb).unsqueeze(1)
        proprio_tokens = torch.stack(
            [proj(x) for proj, x in zip(self.proprio_proj, proprio.split(self.proprio_split, dim=-1))], dim=1
        ) + self.proprio_emb
        tokens = torch.cat([self.cls_token.expand(n, -1, -1), kp_tokens, obj_token, proprio_tokens], dim=1)
        visible = torch.zeros(n, 1, dtype=torch.bool, device=kp.device)
        rest = torch.zeros(n, 1 + len(self.proprio_split), dtype=torch.bool, device=kp.device)
        out = self.transformer(tokens, src_key_padding_mask=torch.cat([visible, kp_mask, rest], dim=1))
        z = self.mu_head(out[:, 0])
        return F.normalize(z, dim=-1, eps=1e-8) if self.latent_normalize else z


class _KpRollout(nn.Module):
    def __init__(self, policy: "LatentKpDistillPolicy", mode: str):
        super().__init__()
        object.__setattr__(self, "policy", policy)
        self.mode = mode
        self.in_keys = [CMD_KEY, OBS_KEY, KP_KEY, "is_init"]
        self.out_keys = [ACTION_KEY] if mode == "deploy" else [ACTION_KEY, KP_MASK_KEY, KP_MODE_KEY, Z_FULL_KEY, TEACHER_ACTION_KEY]

    @torch.no_grad()
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        p = self.policy
        if self.mode == "train":
            p.kp_vecnorm(tensordict[KP_KEY])  # running stats: rollout only, never in train_op
        kp_mask, kp_mode = p.update_masks(tensordict, self.mode)
        x = p.prepare_inputs(tensordict)
        student = p.unwrapped_encoder()
        z_kp = student(x["kp"], x["obj"], x["proprio"], kp_mask)
        action = p.stage1.decode(z_kp, x["proprio"])
        if self.mode == "deploy":
            tensordict.set(ACTION_KEY, action)
            return tensordict
        z_full = p.stage1.encoder(x["goal"], x["proprio"], torch.zeros(x["goal"].shape[0], p.num_goal_tokens, dtype=torch.bool, device=p.device))
        teacher_action = p.stage1.decode(z_full, x["proprio"])
        tensordict.set(ACTION_KEY, teacher_action if p.cfg.rollout_actor == "stage1" else action)
        tensordict.set(KP_MASK_KEY, kp_mask.clone())
        tensordict.set(KP_MODE_KEY, kp_mode.clone())
        tensordict.set(Z_FULL_KEY, z_full)
        tensordict.set(TEACHER_ACTION_KEY, teacher_action)
        return tensordict


class LatentKpDistillPolicy(PPOBase):
    requires_rollout_value = False

    @classmethod
    def from_env(cls, cfg, env, device):
        return cls(cfg, env.observation_spec, env.action_spec, env.reward_spec, device, getattr(env, "base_env", env))

    def __init__(self, cfg: LatentKpDistillConfig, observation_spec, action_spec, reward_spec, device, env):
        super().__init__()
        self.cfg = cfg if isinstance(cfg, LatentKpDistillConfig) else LatentKpDistillConfig(**dict(cfg))
        if self.cfg.stage1_checkpoint is None:
            raise ValueError("algo.stage1_checkpoint is required")
        self.device = device
        object.__setattr__(self, "env", env)

        # --- layouts from the env ---
        goal_group = env.observation_groups[CMD_KEY]
        proprio_group = env.observation_groups[OBS_KEY]
        kp_group = env.observation_groups[KP_KEY]
        goal_names = list(goal_group.keys())
        self.num_goal_tokens = len(goal_names)
        obj_idx = goal_names.index(self.cfg.object_term)
        starts = [0]
        for s in goal_group.split:
            starts.append(starts[-1] + s)
        self.obj_slice = slice(starts[obj_idx], starts[obj_idx + 1])

        if len(kp_group.split) != 1:
            raise ValueError(f"{KP_KEY} must hold exactly one term, got {list(kp_group.keys())}")
        kp_term = kp_group[list(kp_group.keys())[0]]
        # Index space depends on the term class (obs_body_names vs tracking_body_names).
        all_names = list(getattr(kp_term.command_manager, kp_term.available_body_names_attr))
        self.body_names = [all_names[i] for i in kp_term.body_indices_tracking.tolist()]
        self.num_steps = len(kp_term.future_step_indices)
        if kp_group.split[0] != self.num_steps * len(self.body_names) * 3:
            raise ValueError(f"unexpected {KP_KEY} size {kp_group.split[0]}")
        print(f"[latent_kp_distill] keypoint bodies (env order): {self.body_names}, future steps: {self.num_steps}")

        self.mode_names = list(self.cfg.modes)
        self.bernoulli_idx = self.mode_names.index(self.cfg.bernoulli_mode)
        vis = torch.zeros(len(self.mode_names), len(self.body_names), dtype=torch.bool)
        for i, m in enumerate(self.mode_names):
            for b in self.cfg.modes[m]:
                if b not in self.body_names:
                    raise ValueError(f"mode {m!r} body {b!r} not among keypoint bodies {self.body_names}")
                vis[i, self.body_names.index(b)] = True
        self.mode_visible = vis.to(device)

        # --- frozen Stage 1 ---
        ckpt = parse_checkpoint(self.cfg.stage1_checkpoint)
        ckpt.update()
        path = ckpt.get_path()
        print(f"[latent_kp_distill] stage1 checkpoint: {self.cfg.stage1_checkpoint} -> {path}")
        s1 = torch.load(path, map_location="cpu", weights_only=False)
        s1_algo = OmegaConf.create(s1["cfg"]).algo
        if s1_algo.name != "hdmi_latent_distill":
            raise ValueError(f"stage1_checkpoint was trained with algo {s1_algo.name!r}, expected hdmi_latent_distill")
        s1_policy = s1["policy"]
        if list(s1_policy["goal_term_names"]) != goal_names:
            raise ValueError(f"goal terms differ from Stage 1: {s1_policy['goal_term_names']} vs {goal_names}")
        obs_dims = {k: observation_spec[k].shape[-1] for k in (CMD_KEY, OBS_KEY)}
        action_dim = env.action_manager.action_dim
        self.teacher = FrozenTeacher(s1_policy["teacher_checkpoint"], action_dim, obs_dims, device)
        enc1 = TermTokenEncoder(
            goal_group.split, proprio_group.split, s1_algo.latent_dim, s1_algo.d_model,
            s1_algo.nhead, s1_algo.num_layers, s1_algo.ffn_dim, s1_algo.latent_normalize,
        )
        self.stage1 = LatentStudent(enc1, sum(proprio_group.split), action_dim, s1_algo.latent_dim, s1_algo.decoder_hidden_dims)
        self.stage1.load_state_dict(s1_policy["student"])
        self.stage1.to(device).requires_grad_(False).eval()

        # --- trainable keypoint encoder ---
        self.kp_vecnorm = VecNorm(input_shape=(kp_group.split[0],), decay=0.9999).to(device)
        self.kp_encoder = KeypointEncoder(enc1, len(self.body_names), self.num_steps * 3, obj_idx, s1_algo.d_model).to(device)
        if aa.is_distributed():
            self.kp_encoder = DDP(self.kp_encoder, device_ids=[aa.get_local_rank()], broadcast_buffers=False)
        self.opt = torch.optim.Adam(self.kp_encoder.parameters(), lr=self.cfg.lr)

        n = env.num_envs
        self._mask = torch.zeros(n, len(self.body_names), dtype=torch.bool, device=device)
        self._mode = torch.zeros(n, dtype=torch.long, device=device)
        self._phase_seen = -1

    # ---------------------------------------------------------------- helpers
    def unwrapped_encoder(self) -> KeypointEncoder:
        return self.kp_encoder.module if isinstance(self.kp_encoder, DDP) else self.kp_encoder

    def _iter(self) -> int:
        return int(getattr(self.env, "current_iter", 0))

    def curriculum(self) -> tuple[int, torch.Tensor, float]:
        """(phase, mode probabilities, bernoulli p_see) at the current iteration."""
        c, it = self.cfg, self._iter()
        n_modes = len(self.mode_names)
        bern_only = torch.zeros(n_modes, device=self.device)
        bern_only[self.bernoulli_idx] = 1.0
        if it < c.phase1_end_iter:
            return 0, bern_only, 1.0
        if it < c.phase2_end_iter:
            frac = (it - c.phase1_end_iter) / max(c.phase2_end_iter - c.phase1_end_iter, 1)
            return 1, bern_only, 1.0 + frac * (c.p_see_final - 1.0)
        return 2, torch.full((n_modes,), 1.0 / n_modes, device=self.device), c.p_see_final

    def _sample(self, env_ids: torch.Tensor, probs: torch.Tensor, p_see: float) -> None:
        modes = torch.multinomial(probs, env_ids.numel(), replacement=True)
        visible = self.mode_visible[modes].clone()
        bern = modes == self.bernoulli_idx
        if bern.any():
            draw = torch.rand(int(bern.sum()), len(self.body_names), device=self.device) < p_see
            visible[bern] = visible[bern] & draw
        self._mode[env_ids] = modes
        self._mask[env_ids] = ~visible

    def update_masks(self, tensordict: TensorDictBase, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Resample masks for envs that just reset; resample all when the curriculum phase changes."""
        if mode == "train":
            phase, probs, p_see = self.curriculum()
        else:
            probs = torch.zeros(len(self.mode_names), device=self.device)
            probs[self.mode_names.index(self.cfg.eval_mode)] = 1.0
            phase, p_see = -2, self.cfg.eval_p_see
        if phase != self._phase_seen:
            reset = torch.ones(self._mask.shape[0], dtype=torch.bool, device=self.device)
            self._phase_seen = phase
        else:
            reset = tensordict["is_init"].reshape(-1).bool()
        if reset.any():
            self._sample(reset.nonzero().squeeze(-1), probs, p_see)
        return self._mask, self._mode

    def prepare_inputs(self, tensordict: TensorDictBase) -> dict[str, torch.Tensor]:
        goal = self.teacher.normalize(CMD_KEY, tensordict[CMD_KEY])
        proprio = self.teacher.normalize(OBS_KEY, tensordict[OBS_KEY])
        kp = self.kp_vecnorm._normalize(tensordict[KP_KEY])
        # Obs layout is [step, body, xyz]; one token per body holds all its steps.
        lead = kp.shape[:-1]
        kp = kp.reshape(*lead, self.num_steps, len(self.body_names), 3).transpose(-3, -2).reshape(*lead, len(self.body_names), self.num_steps * 3)
        return {"goal": goal, "proprio": proprio, "kp": kp, "obj": goal[..., self.obj_slice]}

    # ------------------------------------------------------------- interface
    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        return _KpRollout(self, mode)

    def get_next_saved_keys(self):
        return ()

    def train_op(self, tensordict: TensorDictBase) -> dict:
        cfg = self.cfg
        x = self.prepare_inputs(tensordict)
        z_full = tensordict[Z_FULL_KEY]
        teacher_action = tensordict[TEACHER_ACTION_KEY]
        kp_mask = tensordict[KP_MASK_KEY].bool()
        kp_mode = tensordict[KP_MODE_KEY].long()
        T = tensordict.shape[1]

        n_modes = len(self.mode_names)
        lat_sum = torch.zeros(n_modes, device=self.device)
        cnt = torch.zeros(n_modes, device=self.device)
        sums = dict(latent=0.0, behavior=0.0, grad_norm=0.0)
        n_steps = n_opt = 0
        for epoch in range(cfg.num_epochs):
            acc = None
            for t in range(T):
                z_kp = self.kp_encoder(x["kp"][:, t], x["obj"][:, t], x["proprio"][:, t], kp_mask[:, t])
                lat = 1.0 - F.cosine_similarity(z_kp, z_full[:, t], dim=-1)
                action = self.stage1.decode(z_kp, x["proprio"][:, t])
                beh = (action - teacher_action[:, t]).square().mean()
                loss = cfg.weight_latent * lat.mean() + cfg.weight_behavior * beh
                acc = loss if acc is None else acc + loss
                if epoch == cfg.num_epochs - 1:
                    with torch.no_grad():
                        lat_sum.index_add_(0, kp_mode[:, t], lat.detach())
                        cnt.index_add_(0, kp_mode[:, t], torch.ones_like(lat))
                        sums["latent"] += lat.mean().item()
                        sums["behavior"] += beh.item()
                        n_steps += 1
                if (t + 1) % cfg.gradient_length == 0 or t == T - 1:
                    self.opt.zero_grad(set_to_none=True)
                    acc.backward()
                    sums["grad_norm"] += nn.utils.clip_grad_norm_(self.kp_encoder.parameters(), cfg.max_grad_norm).item()
                    self.opt.step()
                    n_opt += 1
                    acc = None

        self.num_updates += 1
        phase, _, p_see = self.curriculum()
        info = {
            "kp/latent_loss": sums["latent"] / max(n_steps, 1),
            "kp/behavior_loss": sums["behavior"] / max(n_steps, 1),
            "kp/grad_norm": sums["grad_norm"] / max(n_opt, 1),
            "kp/phase": float(phase),
            "kp/p_see": p_see,
            "kp/visible_frac": (~kp_mask).float().mean().item(),
        }
        if aa.is_distributed():
            dist.all_reduce(lat_sum)
            dist.all_reduce(cnt)
            keys = sorted(info)
            vals = torch.tensor([info[k] for k in keys], device=self.device)
            dist.all_reduce(vals, op=dist.ReduceOp.AVG)
            info = dict(zip(keys, vals.tolist()))
        for i, m in enumerate(self.mode_names):
            if cnt[i] > 0:
                info[f"kp/latent_loss_{m}"] = (lat_sum[i] / cnt[i]).item()
        return info

    def state_dict(self):
        state = OrderedDict()
        state["kp_encoder"] = self.unwrapped_encoder().state_dict()
        state["kp_vecnorm"] = self.kp_vecnorm.state_dict()
        state["opt"] = self.opt.state_dict()
        state["stage1_checkpoint"] = self.cfg.stage1_checkpoint
        state["body_names"] = self.body_names
        state["mode_names"] = self.mode_names
        state["last_iter"] = self._iter()
        return state

    def load_state_dict(self, state_dict, strict=True):
        if "kp_encoder" not in state_dict:
            raise KeyError(
                "Checkpoint has no 'kp_encoder'. Pass Stage 1 via algo.stage1_checkpoint; "
                "checkpoint_path is only for resuming a latent_kp_distill run."
            )
        if list(state_dict.get("body_names", self.body_names)) != self.body_names:
            raise ValueError(f"keypoint bodies differ: {state_dict['body_names']} vs {self.body_names}")
        self.unwrapped_encoder().load_state_dict(state_dict["kp_encoder"], strict=strict)
        self.kp_vecnorm.load_state_dict(state_dict["kp_vecnorm"])
        if "opt" in state_dict:
            self.opt.load_state_dict(state_dict["opt"])
        if state_dict.get("stage1_checkpoint") != self.cfg.stage1_checkpoint:
            warnings.warn(
                f"Resuming with Stage 1 {self.cfg.stage1_checkpoint!r}, "
                f"checkpoint was trained against {state_dict.get('stage1_checkpoint')!r}"
            )
        if hasattr(self.env, "set_progress"):
            self.env.set_progress(state_dict.get("last_iter", 0))
        return []
