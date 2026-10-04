from __future__ import annotations

import torch
from mimic_lite.tasks.deferred import DeferredObservation as BaseObservation
from mimic_lite.tasks.observations.track import _tracking_body_future_observation
from mimic_lite.tasks.transforms import _body_pose_in_anchor_frame

from active_adaptation.utils.math import matrix_from_quat, quat_conjugate, quat_mul, quat_rotate

from .command import RobotObjectTracking

ObjectObservation = BaseObservation[RobotObjectTracking]


class _single_object_observation(ObjectObservation):
    def _initialize_impl(self) -> None:
        names = self.command_manager.object_tracking_body_names
        if len(names) != 1:
            raise ValueError(
                "Phase 1 object observations require exactly one tracked body, "
                f"got {names}"
            )
        self.object_index = self.command_manager.tracking_body_names.index(names[0])


class object_pose_local(_single_object_observation, namespace="hdmi"):
    """Actual object pose in the actual robot projected-yaw anchor."""

    def compute(self) -> torch.Tensor:
        position = self.command_manager.robot_body_pos_local[:, self.object_index]
        rotation = matrix_from_quat(
            self.command_manager.robot_body_quat_local[:, self.object_index]
        )
        return torch.cat([position, rotation[:, :2, :].reshape(self.num_envs, 6)], -1)


def _transform_object_points(
    points: torch.Tensor,
    position: torch.Tensor,
    quaternion: torch.Tensor,
) -> torch.Tensor:
    return quat_rotate(quaternion[:, None, :], points) + position[:, None, :]


class object_surface_points_local(_single_object_observation, namespace="hdmi"):
    """Fixed object-surface points in the actual projected-yaw robot frame."""

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        command = self.command_manager
        if command.object_surface_points is None or command.object_variant_ids is None:
            raise ValueError("object_surface_points_local requires object mesh variants")

    def compute(self) -> torch.Tensor:
        command = self.command_manager
        points = command.object_surface_points[command.object_variant_ids]
        position = command.robot_body_pos_local[:, self.object_index]
        quaternion = command.robot_body_quat_local[:, self.object_index]
        return _transform_object_points(points, position, quaternion).flatten(1)


class object_category(ObjectObservation, namespace="hdmi"):
    """Fixed per-environment object variant ID for category metrics."""

    def _initialize_impl(self) -> None:
        if self.command_manager.object_variant_ids is None:
            raise ValueError("object_category requires object mesh variants")

    def compute(self) -> torch.Tensor:
        return self.command_manager.object_variant_ids[:, None].float()


class object_motion_progress(ObjectObservation, namespace="hdmi"):
    """Reference-motion progress used only for per-category diagnostics."""

    def compute(self) -> torch.Tensor:
        command = self.command_manager
        return (command.t.float() / command.motion_len.clamp_min(1)).unsqueeze(-1)


class ref_body_pos_future_robot_anchor(_tracking_body_future_observation, namespace="hdmi"):
    """Reference body positions at the command's future steps, in the *robot's* current
    projected-yaw anchor frame.

    ``mimic_lite.ref_body_pos_future_local`` expresses the same bodies relative to the
    *reference's* own anchor, so it carries no information about where the robot is: a policy
    reading only it cannot see or correct drift. This term does (AnyBody keypoints are likewise
    in the robot-anchor frame). Layout: [step, body, xyz], flattened.
    """

    def compute(self) -> torch.Tensor:
        command = self.command_manager
        position_w = self._select_body_future(command.ref_body_pos_future_w)
        quaternion_w = self._select_body_future(command.ref_body_quat_future_w)
        position, _ = _body_pose_in_anchor_frame(
            command.robot_anchor_pos_w[:, None, None],
            command.robot_anchor_quat_w[:, None, None],
            position_w,
            quaternion_w,
        )
        return position.reshape(self.num_envs, -1)


# --- Reward-only observations for latent RL (noise-free; kept in a trailing-underscore group) ---


class kp_pos_error_w(_tracking_body_future_observation, namespace="hdmi"):
    """Reference minus robot body position at the current step, world frame. [N, B*3]."""

    def compute(self) -> torch.Tensor:
        command = self.command_manager
        current = command.obs_current_step_index
        ref = command.ref_body_pos_future_w[:, current].index_select(1, self.body_indices_tracking)
        robot = command.robot_body_link_pos_w.index_select(1, self.body_indices_tracking)
        return (ref - robot).reshape(self.num_envs, -1)


class anchor_error_w(_tracking_body_future_observation, namespace="hdmi"):
    """Anchor (pelvis) world position error (3) and orientation error angle in rad (1). [N, 4]."""

    def compute(self) -> torch.Tensor:
        command = self.command_manager
        current = command.obs_current_step_index
        position = command.ref_anchor_pos_future_w[:, current] - command.robot_anchor_pos_w
        relative = quat_mul(quat_conjugate(command.robot_anchor_quat_w), command.ref_anchor_quat_future_w[:, current])
        angle = 2.0 * torch.acos(relative[:, 0].abs().clamp(max=1.0))
        return torch.cat([position, angle[:, None]], dim=-1)
