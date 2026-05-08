

import torch


from mjlab.envs import ManagerBasedRlEnv


from .observations import filtered_force_torque


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

    failure: torch.Tensor = command.metrics["failure"]
    failure = failure.bool()
    return failure


def ft_exceed_limit(
    env: ManagerBasedRlEnv,
    force_limit: float = 500.0,
    torque_limit: float = 50.0,
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
    ft = filtered_force_torque(env)

    # Separate force and torque
    force = ft[..., :3]
    torque = ft[..., 3:]

    # Compute magnitudes
    force_mag = torch.norm(force, dim=-1)
    torque_mag = torch.norm(torque, dim=-1)

    # Check if either exceeds limit
    exceed = torch.logical_or(force_mag > force_limit, torque_mag > torque_limit)

    return exceed