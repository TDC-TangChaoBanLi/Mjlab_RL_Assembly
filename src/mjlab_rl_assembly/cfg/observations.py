

import torch


from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul


from mjlab_rl_assembly.cfg.constants import EE_SITE_NAME, PEG_SITE_NAME, UR5E_ENTITY_NAME, PEG_ENTITY_NAME
from .commands import ReachTargetCommand

import mujoco


def filtered_force_torque(
    env: ManagerBasedRlEnv,
    force_sensor_name: str = "ur_ft_frame_SENSOR_FORCE",
    torque_sensor_name: str = "ur_ft_frame_SENSOR_TORQUE",
    alpha: float = 0.2,
) -> torch.Tensor:
    """Read force and torque sensors from MuJoCo, apply EWMA filtering per env.

    Returns (num_envs, 6) tensor: [fx, fy, fz, tx, ty, tz]
    """
    mj_model = env.sim.mj_model
    mj_data = env.sim.mj_data

    # get sensor ids
    fid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, force_sensor_name)
    tid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, torque_sensor_name)

    if fid < 0 or tid < 0:
        # return zeros if sensors not found
        return torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32)

    # sensor addresses and dims
    f_adr = int(mj_model.sensor_adr[fid])
    f_dim = int(mj_model.sensor_dim[fid])
    t_adr = int(mj_model.sensor_adr[tid])
    t_dim = int(mj_model.sensor_dim[tid])

    # read raw sensor data from mj_data.sensordata (flat array)
    # assume sensors give 3 dims each
    raw_force = torch.tensor(mj_data.sensordata[f_adr : f_adr + f_dim], device=env.device, dtype=torch.float32)
    raw_torque = torch.tensor(mj_data.sensordata[t_adr : t_adr + t_dim], device=env.device, dtype=torch.float32)

    # sensordata is global for whole sim; assume sensor data per-env packaged elsewhere.
    # For single-env setups this returns a single 3-vector each.
    raw = torch.cat([raw_force, raw_torque], dim=0).unsqueeze(0)  # (1,6)

    # caching EWMA state on env.scene (per-sensor key)
    key = f"_ewma_ft_{force_sensor_name}_{torque_sensor_name}"
    if not hasattr(env.scene, "_obs_state"):
        env.scene._obs_state = {}

    if key not in env.scene._obs_state:
        env.scene._obs_state[key] = raw.clone()

    prev = env.scene._obs_state[key]
    updated = alpha * raw + (1.0 - alpha) * prev
    env.scene._obs_state[key] = updated

    # Expand/replicate to num_envs if necessary
    if updated.shape[0] == 1 and env.num_envs > 1:
        updated = updated.expand(env.num_envs, -1)

    return updated.to(device=env.device, dtype=torch.float32)



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
        raise ValueError(
            f"EE site '{EE_SITE_NAME}' not found in entity '{asset_cfg.name}'."
        )

    ee_site_id = site_ids[0]

    # --------------------------------------------------
    # 1. 当前末端位姿
    # --------------------------------------------------
    ee_pos_w = robot.data.site_pos_w[:, ee_site_id]
    ee_quat_w = robot.data.site_quat_w[:, ee_site_id]

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