

import torch


from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.observations import joint_pos_rel
from mjlab.managers.scene_entity_config import SceneEntityCfg


from .commands import ReachTargetCommand


def success_peg_in_hole(
    env: ManagerBasedRlEnv,
    command_name: str
) -> torch.Tensor:
    """
    成功终止条件：
    UR_EE_SITE 与 PEG_SITE 的位置误差和姿态误差均进入容差。
    """
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, ReachTargetCommand):
        raise TypeError(
            f"Command '{command_name}' must be a ReachTargetCommand, got {type(command)}"
        )

    success: torch.Tensor = command.metrics["success"]
    success = success.bool()

    return success


def failure_peg_in_hole(
    env: ManagerBasedRlEnv,
    command_name: str
) -> torch.Tensor:
    """
    失败终止条件：
    UR_EE_SITE 与 PEG_SITE 的位置误差或姿态误差超出容差。
    """
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, ReachTargetCommand):
        raise TypeError(
            f"Command '{command_name}' must be a ReachTargetCommand, got {type(command)}"
        )

    failure: torch.Tensor = command.metrics["failure_pose"]
    failure = failure.bool()
    return failure


def ft_exceed_limit(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """
    失败终止条件：
    力或力矩传感器的值超过设定的阈值。

    Args:
        env: The environment
        force_limit: Maximum allowed force magnitude (N)
        torque_limit: Maximum allowed torque magnitude (N*m)

    Returns:
        Tensor of shape (num_envs,): True if force or torque exceeds limit
    """
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, ReachTargetCommand):
        raise TypeError(
            f"Command '{command_name}' must be a ReachTargetCommand, got {type(command)}"
        )
    
    failure: torch.Tensor = command.metrics["failure_ft"]
    failure = failure.bool()

    return failure


def joint_pos_rel_has_nan(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    失败终止条件：
    joint_pos_rel 返回的关节位置观测中包含 NaN 或 Inf。

    当关节位置观测异常时终止 episode，避免训练数据污染。

    Args:
        env: The environment
        asset_cfg: Scene entity configuration for the robot

    Returns:
        Tensor of shape (num_envs,): True if joint_pos_rel contains NaN/Inf
    """
    obs = joint_pos_rel(env, asset_cfg=asset_cfg)
    has_nan = ~torch.isfinite(obs).all(dim=-1)
    return has_nan.bool()