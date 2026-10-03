"""AnyBody Stage 1 on HDMI: distill an HDMI teacher into a latent-bottleneck student.

Port of AnyBody's MUSE-Transformer distillation (``LatentBottleneckMUSETransformer`` +
``MuseDistillation``) onto the active-adaptation training loop.

    teacher:  a = MLP(policy, command)                                  # frozen mimic_lite_ppo actor
    student:  z = normalize(E(command tokens, policy tokens))           # transformer, [CLS] -> z
              a = D(z, policy)                                          # MLP decoder = motor prior
    loss:     ||a_student - a_teacher||^2 + w * (1 - cos(z_t, z_{t-1}))  # DAgger: student acts

Differences from AnyBody, all deliberate:
- Tokens are per observation *term* (read from the env's ObsGroup layout) instead of per proprio
  frame. HDMI groups mix history lengths (7-frame history, 3-step prev action, 1-step object pose),
  so frames are not a uniform axis; terms are, and they give Stage 2 a natural masking unit.
- The student reuses the teacher's frozen VecNorm statistics instead of learning its own, so both
  see identical inputs and the student cannot drift its normalizer away from the teacher's.
- The goal mask is a per-goal-token boolean ``[N, num_goal_tokens]``. Stage 1 masks all goal tokens
  together (AnyBody's whole-goal-block mask); Stage 2 can mask individual terms.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence, Tuple
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tensordict import TensorDict, TensorDictBase
from torch.nn.parallel import DistributedDataParallel as DDP

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.learning.ppo.common import ACTION_KEY, CMD_KEY, DONE_KEY, OBS_KEY, make_mlp
from active_adaptation.learning.ppo.ppo_base import PPOBase
from active_adaptation.utils.wandb import parse_checkpoint
from mimic_lite_learning.common import ActorROA


TEACHER_ACTION_KEY = "teacher_action"
GOAL_MASK_KEY = "goal_mask"


@dataclass
class LatentDistillConfig:
    _target_: str = f"{__package__}.latent_distill.LatentDistillConfig"
    name: str = "hdmi_latent_distill"

    # ``run:<entity>/<project>/<run_id>[:<iter>]`` or a local path to a mimic_lite_ppo checkpoint.
    teacher_checkpoint: str | None = None

    train_every: int = 24
    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY)
    goal_key: str = CMD_KEY
    proprio_key: str = OBS_KEY

    # Student (AnyBody G1FlatMUSETransformerDistillationRunnerCfg).
    latent_dim: int = 16
    d_model: int = 192
    nhead: int = 4
    num_layers: int = 2
    ffn_dim: int = 768
    decoder_hidden_dims: Tuple[int, ...] = (1024, 512, 256, 128)
    latent_normalize: bool = True

    # Optimization (AnyBody MuseDistillation defaults + runner overrides).
    lr: float = 1e-3
    max_grad_norm: float = 1.0
    gradient_length: int = 15  # timesteps of loss accumulated per optimizer step
    smoothness_weight: float = 0.1  # cosine smoothness on consecutive z

    # Goal-mask curriculum: p linearly ramps p_start -> p_end over [start_iter, end_iter].
    # Resampled independently per env per step, as in AnyBody's ``_resample_goal_mask``.
    goal_mask_p_start: float = 0.0
    goal_mask_p_end: float = 0.5
    goal_mask_ramp_start_iter: int = 500
    goal_mask_ramp_end_iter: int = 4000
    eval_goal_mask_p: float = 0.0

    # Who drives the env during training rollouts. "student" is DAgger (AnyBody). "teacher" is a
    # diagnostic: episode stats then measure the rebuilt teacher, which should match its own run.
    rollout_actor: str = "student"

    def __post_init__(self):
        self.in_keys = tuple(self.in_keys)
        self.decoder_hidden_dims = tuple(self.decoder_hidden_dims)
        if self.goal_key not in self.in_keys or self.proprio_key not in self.in_keys:
            raise ValueError(
                f"in_keys {self.in_keys} must contain goal_key={self.goal_key!r} "
                f"and proprio_key={self.proprio_key!r}"
            )
        if self.rollout_actor not in ("student", "teacher"):
            raise ValueError(f"rollout_actor must be 'student' or 'teacher', got {self.rollout_actor!r}")
        if self.gradient_length < 1:
            raise ValueError("gradient_length must be >= 1")
        if self.goal_mask_ramp_end_iter < self.goal_mask_ramp_start_iter:
            raise ValueError("goal_mask_ramp_end_iter must be >= goal_mask_ramp_start_iter")

    def get_class(self):
        return LatentDistillPolicy


cs = ConfigStore.instance()
cs.store("hdmi_latent_distill", node=LatentDistillConfig, group="algo")


class TermTokenEncoder(nn.Module):
    """[CLS] + one token per goal term + one token per proprio term -> transformer -> z."""

    def __init__(
        self,
        goal_split: Sequence[int],
        proprio_split: Sequence[int],
        latent_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ffn_dim: int,
        latent_normalize: bool,
    ):
        super().__init__()
        self.goal_split = tuple(int(s) for s in goal_split)
        self.proprio_split = tuple(int(s) for s in proprio_split)
        self.num_goal_tokens = len(self.goal_split)
        self.latent_normalize = latent_normalize

        self.goal_proj = nn.ModuleList(nn.Linear(s, d_model) for s in self.goal_split)
        self.proprio_proj = nn.ModuleList(nn.Linear(s, d_model) for s in self.proprio_split)
        # One learned identity embedding per term token; replaces AnyBody's positional + modality
        # embeddings since every token here is a distinct term.
        num_tokens = len(self.goal_split) + len(self.proprio_split)
        self.token_emb = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.mu_head = nn.Linear(d_model, latent_dim)

    def forward(
        self, goal: torch.Tensor, proprio: torch.Tensor, goal_mask: torch.Tensor
    ) -> torch.Tensor:
        """goal [N, G], proprio [N, P], goal_mask [N, num_goal_tokens] (True = hidden) -> z [N, L]."""
        goal_terms = goal.split(self.goal_split, dim=-1)
        proprio_terms = proprio.split(self.proprio_split, dim=-1)
        tokens = [proj(x) for proj, x in zip(self.goal_proj, goal_terms)]
        tokens += [proj(x) for proj, x in zip(self.proprio_proj, proprio_terms)]
        tokens = torch.stack(tokens, dim=-2) + self.token_emb

        # Zero masked goal tokens as well as padding them out of attention, so a masked token's
        # K/V can never leak goal information even if the padding mask were misapplied.
        goal_tokens = tokens[:, : self.num_goal_tokens]
        goal_tokens = goal_tokens.masked_fill(goal_mask.unsqueeze(-1), 0.0)
        tokens = torch.cat(
            [self.cls_token.expand(tokens.shape[0], -1, -1), goal_tokens, tokens[:, self.num_goal_tokens :]],
            dim=-2,
        )
        n = tokens.shape[0]
        visible = torch.zeros(n, 1, dtype=torch.bool, device=tokens.device)
        proprio_pad = torch.zeros(n, len(self.proprio_split), dtype=torch.bool, device=tokens.device)
        key_padding_mask = torch.cat([visible, goal_mask, proprio_pad], dim=-1)

        out = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        z = self.mu_head(out[:, 0])
        if self.latent_normalize:
            z = F.normalize(z, dim=-1, eps=1e-8)
        return z


class LatentStudent(nn.Module):
    def __init__(self, encoder: TermTokenEncoder, proprio_dim: int, action_dim: int, latent_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.encoder = encoder
        layers, in_dim = [], latent_dim + proprio_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.decoder = nn.Sequential(*layers)

    def decode(self, z: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, proprio], dim=-1))

    def forward(
        self, goal: torch.Tensor, proprio: torch.Tensor, goal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(goal, proprio, goal_mask)
        return self.decode(z, proprio), z


class FrozenTeacher(nn.Module):
    """Deterministic mean action of a mimic_lite_ppo actor, rebuilt from its checkpoint."""

    def __init__(self, checkpoint_spec: str, action_dim: int, obs_dims: dict[str, int], device):
        super().__init__()
        ckpt = parse_checkpoint(checkpoint_spec)
        ckpt.update()
        path = ckpt.get_path()
        print(f"[latent_distill] teacher checkpoint: {checkpoint_spec} -> {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        algo = OmegaConf.create(state["cfg"]).algo
        if algo.get("use_actor_encoder", False) or algo.get("res_actor_hidden_dims"):
            raise NotImplementedError(
                "Only plain-MLP mimic_lite_ppo teachers are supported "
                "(use_actor_encoder=False, no residual actor)."
            )
        self.in_keys = tuple(algo.actor_in_keys)

        self.vecnorms = nn.ModuleDict()
        for key in self.in_keys:
            vecnorm = VecNorm(input_shape=(obs_dims[key],), decay=0.9999)
            vecnorm.load_state_dict(
                {k.split(".", 1)[1]: v for k, v in state["policy"]["vecnorms"].items() if k.startswith(f"{key}.")}
            )
            self.vecnorms[key] = vecnorm

        self.mlp = make_mlp(list(algo.actor_hidden_dims), norm=algo.layer_norm)
        self.head = ActorROA(action_dim)
        self.mlp(torch.zeros(1, sum(obs_dims[k] for k in self.in_keys)))
        self.head(torch.zeros(1, algo.actor_hidden_dims[-1]))

        # mimic_lite_ppo actor layout: ProbabilisticActor -> Seq[CatTensors, Mod(mlp), Mod(ActorROA)].
        actor_state = state["policy"]["actor"]
        mlp_prefix, head_prefix = "module.0.module.1.module.", "module.0.module.2.module."
        unexpected = [k for k in actor_state if not k.startswith((mlp_prefix, head_prefix))]
        if unexpected:
            raise RuntimeError(f"Unexpected teacher actor keys (architecture mismatch?): {unexpected}")
        self.mlp.load_state_dict({k[len(mlp_prefix):]: v for k, v in actor_state.items() if k.startswith(mlp_prefix)})
        self.head.load_state_dict({k[len(head_prefix):]: v for k, v in actor_state.items() if k.startswith(head_prefix)})

        self.to(device)
        self.requires_grad_(False)
        self.eval()

    def normalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        # Never update running stats: the teacher's statistics define the input space.
        return self.vecnorms[key]._normalize(x)

    @torch.no_grad()
    def forward(self, normalized: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([normalized[k] for k in self.in_keys], dim=-1)
        loc, _ = self.head(self.mlp(x))
        return loc


class _StudentRollout(nn.Module):
    """Collector-side policy: student acts, teacher labels, goal mask is sampled per env per step."""

    def __init__(self, policy: "LatentDistillPolicy", mode: str):
        super().__init__()
        object.__setattr__(self, "policy", policy)
        self.mode = mode
        self.in_keys = [policy.cfg.goal_key, policy.cfg.proprio_key]
        self.out_keys = [ACTION_KEY] if mode == "deploy" else [ACTION_KEY, TEACHER_ACTION_KEY, GOAL_MASK_KEY]

    @torch.no_grad()
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        p = self.policy
        normalized = p.normalized_inputs(tensordict)
        n = tensordict.shape[0]
        mask_p = p.goal_mask_p() if self.mode == "train" else p.cfg.eval_goal_mask_p
        goal_mask = (torch.rand(n, 1, device=p.device) < mask_p).expand(n, p.num_goal_tokens)
        student = p.student.module if isinstance(p.student, DDP) else p.student
        action, _ = student(normalized[p.cfg.goal_key], normalized[p.cfg.proprio_key], goal_mask)
        if self.mode == "deploy":
            tensordict.set(ACTION_KEY, action)
            return tensordict
        teacher_action = p.teacher(normalized)
        use_teacher = self.mode == "train" and p.cfg.rollout_actor == "teacher"
        tensordict.set(ACTION_KEY, teacher_action if use_teacher else action)
        tensordict.set(TEACHER_ACTION_KEY, teacher_action)
        tensordict.set(GOAL_MASK_KEY, goal_mask)
        return tensordict


class LatentDistillPolicy(PPOBase):
    requires_rollout_value = False

    @classmethod
    def from_env(cls, cfg, env, device):
        return cls(cfg, env.observation_spec, env.action_spec, env.reward_spec, device, getattr(env, "base_env", env))

    def __init__(self, cfg: LatentDistillConfig, observation_spec, action_spec, reward_spec, device, env):
        super().__init__()
        self.cfg = cfg if isinstance(cfg, LatentDistillConfig) else LatentDistillConfig(**dict(cfg))
        if self.cfg.teacher_checkpoint is None:
            raise ValueError("algo.teacher_checkpoint is required (e.g. run:elijahgalahad/mimic_lite/nnds9gg2:4000)")
        self.device = device
        object.__setattr__(self, "env", env)

        # Term layout comes from the env, never from hand-computed dims.
        goal_group = env.observation_groups[self.cfg.goal_key]
        proprio_group = env.observation_groups[self.cfg.proprio_key]
        goal_split, proprio_split = goal_group.split, proprio_group.split
        self.num_goal_tokens = len(goal_split)
        self.goal_term_names = list(goal_group.keys())
        print(f"[latent_distill] goal tokens   {dict(zip(goal_group.keys(), goal_split))}")
        print(f"[latent_distill] proprio tokens {dict(zip(proprio_group.keys(), proprio_split))}")

        obs_dims = {k: observation_spec[k].shape[-1] for k in self.cfg.in_keys}
        action_dim = env.action_manager.action_dim
        self.teacher = FrozenTeacher(self.cfg.teacher_checkpoint, action_dim, obs_dims, device)
        if set(self.teacher.in_keys) != {self.cfg.goal_key, self.cfg.proprio_key}:
            raise ValueError(f"Teacher reads {self.teacher.in_keys}; student reads {self.cfg.in_keys}")

        encoder = TermTokenEncoder(
            goal_split, proprio_split, self.cfg.latent_dim, self.cfg.d_model,
            self.cfg.nhead, self.cfg.num_layers, self.cfg.ffn_dim, self.cfg.latent_normalize,
        )
        self.student = LatentStudent(
            encoder, sum(proprio_split), action_dim, self.cfg.latent_dim, self.cfg.decoder_hidden_dims
        ).to(device)

        if aa.is_distributed():
            self.student = DDP(self.student, device_ids=[aa.get_local_rank()], broadcast_buffers=False)
        self.opt = torch.optim.Adam(self.student.parameters(), lr=self.cfg.lr)

    def normalized_inputs(self, tensordict: TensorDictBase) -> dict[str, torch.Tensor]:
        return {k: self.teacher.normalize(k, tensordict[k]) for k in (self.cfg.goal_key, self.cfg.proprio_key)}

    def goal_mask_p(self) -> float:
        it = int(getattr(self.env, "current_iter", 0))
        c = self.cfg
        if c.goal_mask_ramp_end_iter == c.goal_mask_ramp_start_iter:
            frac = float(it >= c.goal_mask_ramp_end_iter)
        else:
            frac = (it - c.goal_mask_ramp_start_iter) / (c.goal_mask_ramp_end_iter - c.goal_mask_ramp_start_iter)
        frac = min(max(frac, 0.0), 1.0)
        return c.goal_mask_p_start + frac * (c.goal_mask_p_end - c.goal_mask_p_start)

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        return _StudentRollout(self, mode)

    def get_next_saved_keys(self):
        return ()

    def train_op(self, tensordict: TensorDictBase) -> dict:
        """One pass over the rollout in time order, as AnyBody's MuseDistillation.update.

        tensordict: [N, T, ...]. Loss is accumulated over ``gradient_length`` consecutive steps
        per optimizer step. The smoothness pair (z_t, z_{t-1}) is skipped across episode
        boundaries and across optimizer steps (z_{t-1} came from the pre-update weights).
        """
        cfg = self.cfg
        normalized = self.normalized_inputs(tensordict)
        goal, proprio = normalized[cfg.goal_key], normalized[cfg.proprio_key]
        teacher_action = tensordict[TEACHER_ACTION_KEY]
        goal_mask = tensordict[GOAL_MASK_KEY].bool()
        done = tensordict[DONE_KEY].reshape(*tensordict.shape)  # done[:, t]: episode ended after step t
        T = tensordict.shape[1]

        sums = dict(bc=0.0, bc_masked=0.0, bc_visible=0.0, smooth=0.0, grad_norm=0.0)
        n_masked = n_visible = n_smooth = n_opt = 0
        z_norms, z_dim_std = [], []
        acc, prev_z, prev_done = None, None, None
        for t in range(T):
            action, z = self.student(goal[:, t], proprio[:, t], goal_mask[:, t])
            per_env_bc = (action - teacher_action[:, t]).square().mean(-1)
            loss = per_env_bc.mean()
            sums["bc"] += loss.item()

            with torch.no_grad():
                masked = goal_mask[:, t, 0]
                if masked.any():
                    sums["bc_masked"] += per_env_bc[masked].mean().item(); n_masked += 1
                if (~masked).any():
                    sums["bc_visible"] += per_env_bc[~masked].mean().item(); n_visible += 1
                z_norms.append(z.norm(dim=-1).mean().item())
                z_dim_std.append(z.std(dim=0))

            if prev_z is not None:
                valid = (~prev_done).float()
                if valid.sum() > 0:
                    smooth = ((1.0 - F.cosine_similarity(z, prev_z, dim=-1)) * valid).sum() / valid.sum()
                    loss = loss + cfg.smoothness_weight * smooth
                    sums["smooth"] += smooth.item(); n_smooth += 1

            acc = loss if acc is None else acc + loss
            if (t + 1) % cfg.gradient_length == 0 or t == T - 1:
                self.opt.zero_grad(set_to_none=True)
                acc.backward()
                sums["grad_norm"] += nn.utils.clip_grad_norm_(self.student.parameters(), cfg.max_grad_norm).item()
                self.opt.step()
                n_opt += 1
                acc, prev_z, prev_done = None, None, None
            else:
                prev_z, prev_done = z.detach(), done[:, t]

        self.num_updates += 1
        z_dim_std = torch.stack(z_dim_std).mean(0)
        info = {
            "distill/bc_loss": sums["bc"] / T,
            "distill/bc_loss_goal_visible": sums["bc_visible"] / max(n_visible, 1),
            "distill/bc_loss_goal_masked": sums["bc_masked"] / max(n_masked, 1),
            "distill/smoothness": sums["smooth"] / max(n_smooth, 1),
            "distill/grad_norm": sums["grad_norm"] / max(n_opt, 1),
            "distill/goal_mask_p": self.goal_mask_p(),
            "distill/goal_mask_frac": goal_mask[..., 0].float().mean().item(),
            "distill/z_norm": sum(z_norms) / len(z_norms),
            # Anisotropy: max/min per-dim std of z across envs; >> 1 means a few dims dominate.
            "distill/z_dim_std_mean": z_dim_std.mean().item(),
            "distill/z_dim_std_ratio": (z_dim_std.max() / z_dim_std.min().clamp_min(1e-8)).item(),
        }
        if aa.is_distributed():
            keys = sorted(info)
            vals = torch.tensor([info[k] for k in keys], device=self.device)
            dist.all_reduce(vals, op=dist.ReduceOp.AVG)
            info = dict(zip(keys, vals.tolist()))
        return info

    def state_dict(self):
        student = self.student.module if isinstance(self.student, DDP) else self.student
        state = OrderedDict()
        state["student"] = student.state_dict()
        state["opt"] = self.opt.state_dict()
        state["teacher_checkpoint"] = self.cfg.teacher_checkpoint
        state["goal_term_names"] = self.goal_term_names
        state["last_iter"] = int(getattr(self.env, "current_iter", 0))
        return state

    def load_state_dict(self, state_dict, strict=True):
        if "student" not in state_dict:
            raise KeyError(
                "Checkpoint has no 'student' entry. Pass the teacher via algo.teacher_checkpoint, "
                "not checkpoint_path; checkpoint_path is only for resuming a latent_distill run."
            )
        student = self.student.module if isinstance(self.student, DDP) else self.student
        student.load_state_dict(state_dict["student"], strict=strict)
        if "opt" in state_dict:
            self.opt.load_state_dict(state_dict["opt"])
        if state_dict.get("teacher_checkpoint") != self.cfg.teacher_checkpoint:
            warnings.warn(
                f"Resuming with teacher {self.cfg.teacher_checkpoint!r}, "
                f"checkpoint was trained against {state_dict.get('teacher_checkpoint')!r}"
            )
        start_iter = state_dict.get("last_iter", 0)
        if hasattr(self.env, "set_progress"):
            self.env.set_progress(start_iter)
        return []
