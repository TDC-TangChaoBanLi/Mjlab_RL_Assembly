

import torch


from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul


from mjlab_rl_assembly.cfg.constants import EE_SITE_NAME, PEG_SITE_NAME, UR5E_ENTITY_NAME, PEG_ENTITY_NAME
from .commands import ReachTargetCommand


def quat_to_rotvec(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert quaternion to rotation vector.

    Args:
        quat: Tensor of shape (..., 4), in wxyz order.

    Returns:
        rotvec: Tensor of shape (..., 3), axis-angle / rotation vector.
    """
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp_min(eps)

    # 解决 q 和 -q 表示同一姿态的问题，强制 w >= 0
    quat = torch.where(quat[..., :1] < 0.0, -quat, quat)

    w = quat[..., 0:1].clamp(-1.0, 1.0)
    xyz = quat[..., 1:4]

    sin_half_angle = torch.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half_angle, w)

    axis = xyz / sin_half_angle.clamp_min(eps)
    rotvec = axis * angle

    # 小角度时，quat ≈ [1, rotvec / 2]，因此 rotvec ≈ 2 * xyz
    small_angle = sin_half_angle < eps
    rotvec = torch.where(small_angle, 2.0 * xyz, rotvec)

    return rotvec


def target_pose_ee(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(UR5E_ENTITY_NAME),
) -> torch.Tensor:
    """
    Relative goal pose in end-effector frame.

    Returns:
        Tensor of shape (num_envs, 6):
        [target_pos_ee(3), target_rotvec_ee(3)]

    含义：
        target_pos_ee:
            目标点在末端坐标系下的位置误差。

        target_rotvec_ee:
            目标姿态相对于当前末端姿态的旋转向量误差。
            方向表示旋转轴，模长表示旋转角，单位 rad。
    """
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, ReachTargetCommand):
        raise TypeError(
            f"Command '{command_name}' must be a ReachTargetCommand, got {type(command)}"
        )

    robot: Entity = env.scene[asset_cfg.name]
    site_ids, site_names = robot.find_sites(EE_SITE_NAME)

    if len(site_ids) == 0:
        raise ValueError(f"EE site '{EE_SITE_NAME}' not found in entity '{asset_cfg.name}'.")

    if site_names[0] == EE_SITE_NAME:
        ee_site_id = site_ids[0]
    else:
        ee_site_id = 0

    ee_pos_w = robot.data.site_pos_w[:, ee_site_id]
    ee_quat_w = robot.data.site_quat_w[:, ee_site_id]

    # target pose in world frame
    target_pos_w = command.target_pos
    target_quat_w = command.target_quat

    # --------------------------------------------------
    # 1. 位置误差：world frame -> end-effector frame
    # --------------------------------------------------
    err_pos_w = target_pos_w - ee_pos_w
    target_pos_ee = quat_apply(quat_inv(ee_quat_w), err_pos_w)

    # --------------------------------------------------
    # 2. 姿态误差：target pose relative to current EE frame
    # --------------------------------------------------
    # 原来你写的是：
    #     quat_err = quat_mul(target_quat_w, quat_inv(ee_quat_w))
    #
    # 如果观察项叫 target_pose_ee，更推荐写成：
    #     q_err_ee = inv(q_ee_w) * q_target_w
    #
    # 这样表示“目标姿态在当前末端坐标系下的相对姿态”。
    quat_err_ee = quat_mul(quat_inv(ee_quat_w), target_quat_w)

    quat_err_ee = quat_err_ee / torch.norm(
        quat_err_ee,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)

    # 解决四元数符号二义性
    quat_err_ee = torch.where(
        quat_err_ee[:, :1] < 0.0,
        -quat_err_ee,
        quat_err_ee,
    )

    # quaternion error -> rotation vector error
    target_rotvec_ee = quat_to_rotvec(quat_err_ee)

    return torch.cat([target_pos_ee, target_rotvec_ee], dim=-1)