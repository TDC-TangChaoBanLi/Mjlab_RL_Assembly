

from click import command
import torch


from mjlab.entity import Entity
from mjlab.sensor import BuiltinSensor
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul
from mjlab.envs.mdp.observations import joint_pos_rel


from mjlab_rl_assembly.cfg.constants import EE_SITE_NAME, PEG_SITE_NAME, UR5E_ENTITY_NAME, PEG_ENTITY_NAME, FORCE_SENSOR_NAME, TORQUE_SENSOR_NAME
from .commands import ReachTargetCommand

import mujoco


def get_stage(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """
    Get the current stage from the command.

    Returns:
        Tensor of shape (num_envs, 1): stage value (0 for align, 1 for insert)
    """
    command = env.command_manager.get_term(command_name)

    if not isinstance(command, ReachTargetCommand):
        raise TypeError(
            f"Command '{command_name}' must be a ReachTargetCommand, got {type(command)}"
        )

    stage = command.stage.clone().float().unsqueeze(-1)
    return stage


def filtered_force_torque(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Get filtered force and torque sensor data from the reach_target command metrics.

    Args:
        env: The environment
        force_sensor_name: Name of the force sensor (unused)
        torque_sensor_name: Name of the torque sensor (unused)
        alpha: EWMA filter coefficient (unused, filtering is done in command)

    Returns (num_envs, 6) tensor: [fx, fy, fz, tx, ty, tz]
    """
    # Get the reach_target command
    command = env.command_manager.get_term(command_name)
    
    if not isinstance(command, ReachTargetCommand):
        raise TypeError(
            f"Command 'reach_target' must be a ReachTargetCommand, got {type(command)}"
        )
    
    # Return the pre-filtered sensor data from metrics
    return command.metrics["ft_sensor"]



def safe_normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Safely normalize quaternion in wxyz order.

    Args:
        q: Tensor of shape (..., 4)

    Returns:
        Normalized quaternion with shape (..., 4)
    """
    # 先去除 NaN / Inf，避免后续传播
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    norm = torch.linalg.norm(q, dim=-1, keepdim=True)

    # 零四元数回退为单位四元数
    identity = torch.zeros_like(q)
    identity[..., 0] = 1.0

    q_norm = q / norm.clamp_min(eps)
    q_norm = torch.where(norm < eps, identity, q_norm)

    # 解决 q 和 -q 表示同一姿态的问题，强制 w >= 0
    q_norm = torch.where(q_norm[..., :1] < 0.0, -q_norm, q_norm)

    return q_norm


def quat_to_rotvec_safe(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert quaternion to rotation vector safely.

    Args:
        quat: Tensor of shape (..., 4), in wxyz order.

    Returns:
        rotvec: Tensor of shape (..., 3), axis-angle / rotation vector.
    """
    quat = safe_normalize_quat(quat, eps=eps)

    w = quat[..., 0:1].clamp(-1.0, 1.0)
    xyz = quat[..., 1:4]

    sin_half_angle = torch.linalg.norm(xyz, dim=-1, keepdim=True)

    angle = 2.0 * torch.atan2(sin_half_angle, w)

    axis = xyz / sin_half_angle.clamp_min(eps)
    rotvec = axis * angle

    # 小角度近似：quat ≈ [1, rotvec / 2]
    rotvec_small = 2.0 * xyz
    rotvec = torch.where(sin_half_angle < eps, rotvec_small, rotvec)

    # 最后再清理一次，防止极端情况污染观测
    rotvec = torch.nan_to_num(rotvec, nan=0.0, posinf=0.0, neginf=0.0)

    # 姿态误差最大不应超过 pi，做一下限幅更稳
    rotvec = torch.clamp(rotvec, -3.1415926, 3.1415926)

    return rotvec



def target_pose_ee(
    env: ManagerBasedRlEnv,
    command_name: str,
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

    # --------------------------------------------------
    # 1. 当前末端位姿
    # --------------------------------------------------
    ee_pos_w = command.ee_pos
    ee_quat_w = command.ee_quat

    # --------------------------------------------------
    # 2. 目标位姿
    # --------------------------------------------------
    target_pos_w = command.target_pos
    target_quat_w = command.target_quat

    # --------------------------------------------------
    # 3. 防止 NaN / Inf 传播
    # --------------------------------------------------
    ee_pos_w = torch.nan_to_num(
        ee_pos_w,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    target_pos_w = torch.nan_to_num(
        target_pos_w,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    ee_quat_w = safe_normalize_quat(ee_quat_w)
    target_quat_w = safe_normalize_quat(target_quat_w)

    # --------------------------------------------------
    # 4. 位置误差：world frame -> end-effector frame
    # --------------------------------------------------
    err_pos_w = target_pos_w - ee_pos_w

    target_pos_ee = quat_apply(
        quat_inv(ee_quat_w),
        err_pos_w,
    )

    target_pos_ee = torch.nan_to_num(
        target_pos_ee,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # 可选限幅，避免极端 reset 或 command 异常导致观测过大
    target_pos_ee = torch.clamp(target_pos_ee, -2.0, 2.0)

    # --------------------------------------------------
    # 5. 姿态误差：target relative to current EE
    # --------------------------------------------------
    # 表示：
    #     q_err_ee = inv(q_ee_w) * q_target_w
    #
    # 即目标姿态在当前末端坐标系下的相对姿态。
    quat_err_ee = quat_mul(
        quat_inv(ee_quat_w),
        target_quat_w,
    )

    quat_err_ee = safe_normalize_quat(quat_err_ee)

    # quaternion error -> rotation vector error
    target_rotvec_ee = quat_to_rotvec_safe(quat_err_ee)

    obs = torch.cat(
        [
            target_pos_ee,
            target_rotvec_ee,
        ],
        dim=-1,
    )

    return obs


def ee_pose_world(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """
    End-effector pose in world frame.

    Returns:
        Tensor of shape (num_envs, 7):
        [ee_pos_w(3), ee_quat_w(4)]

    含义：
        ee_pos_w:
            末端在世界坐标系下的位置。

        ee_quat_w:
            末端在世界坐标系下的姿态。
    """
    command = env.command_manager.get_term(command_name)

    if not isinstance(command, ReachTargetCommand):
        raise TypeError(
            f"Command '{command_name}' must be a ReachTargetCommand, got {type(command)}"
        )

    ee_pos_w = command.ee_pos
    ee_quat_w = command.ee_quat

    obs = torch.cat(
        [
            ee_pos_w,
            ee_quat_w,
        ],
        dim=-1,
    )

    return obs


def target_pose_world(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """
    Relative goal pose in world frame.

    Returns:
        Tensor of shape (num_envs, 6):
        [target_pos_ee(3), target_rotvec_ee(3)]

    含义：
        target_pos_w:
            目标点在世界坐标系下的位置误差。

        target_rotvec_w:
            目标姿态相对于世界姿态的旋转向量误差。
            方向表示旋转轴，模长表示旋转角，单位 rad。
    """
    command = env.command_manager.get_term(command_name)

    if not isinstance(command, ReachTargetCommand):
        raise TypeError(
            f"Command '{command_name}' must be a ReachTargetCommand, got {type(command)}"
        )

    target_pos_w = command.target_pos
    target_quat_w = command.target_quat

    obs = torch.cat(
        [
            target_pos_w,
            target_quat_w,
        ],
        dim=-1,
    )

    return obs


def joint_pos_rel_with_history(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    带历史缓存的关节位置观测函数。

    当 joint_pos_rel 返回 NaN/Inf 时，使用上次的有效值作为替代，
    避免观测值污染导致训练不稳定。

    Args:
        env: The environment
        asset_cfg: Scene entity configuration for the robot

    Returns:
        Tensor of shape (num_envs, num_joints): Joint positions relative to defaults
    """
    # 获取原始关节位置观测
    obs = joint_pos_rel(env, asset_cfg=asset_cfg)

    # 获取或初始化历史缓存（存储在 env 的自定义属性中）
    if not hasattr(env, '_joint_pos_rel_history'):
        # 初始化时使用当前观测值（假设首次观测是有效的）
        env._joint_pos_rel_history = obs.clone()

    # 检测 NaN/Inf 并使用历史值替换
    finite_mask = torch.isfinite(obs)
    obs = torch.where(finite_mask, obs, env._joint_pos_rel_history)

    # 更新历史缓存为当前有效观测
    env._joint_pos_rel_history = obs.clone()

    return obs