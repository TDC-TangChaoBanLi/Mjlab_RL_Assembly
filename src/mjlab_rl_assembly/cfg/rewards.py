

from matplotlib.pylab import f
import torch


from mjlab.envs import ManagerBasedRlEnv


from .observations import filtered_force_torque


from .commands import ReachTargetCommand


def ft_penalty(
    env: ManagerBasedRlEnv,
    force_sensor_name: str = "ur_ft_frame_SENSOR_FORCE",
    torque_sensor_name: str = "ur_ft_frame_SENSOR_TORQUE",
    alpha: float = 0.2,
    force_std: float = 500.0,
    torque_std: float = 50.0,
) -> torch.Tensor:
    """
    Penalty for force and torque sensor values, normalized by scale factors.

    Args:
        env: The environment
        force_sensor_name: Name of the force sensor
        torque_sensor_name: Name of the torque sensor
        alpha: EWMA filter alpha parameter
        force_std: Standard deviation for force (N)
        torque_std: Standard deviation for torque (N*m)

    Returns:
        Tensor of shape (num_envs,): penalty value (negative for minimization)
    """
    ft = filtered_force_torque(env, force_sensor_name, torque_sensor_name, alpha)

    # Separate force and torque
    force = ft[..., :3]
    torque = ft[..., 3:]

    # Normalize and compute penalty
    force_norm = torch.norm(force, dim=-1)
    torque_norm = torch.norm(torque, dim=-1)

    # Normalize by standard deviation
    force_std = -torch.exp(-(force_norm / force_std)**2) + 1.0
    torque_std = -torch.exp(-(torque_norm / torque_std)**2) + 1.0

    # Combined penalty (negative for reward minimization)
    penalty = (force_std + torque_std)/2.0

    return penalty



def pos_reach_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """
    Gaussian reward for reaching target position.
    
    Args:
        env: The environment
        command_name: Name of the command term
        std: Standard deviation for the Gaussian distribution

    Returns:
        Tensor of shape (num_envs,): reward value (positive for maximization)
    """
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, ReachTargetCommand):
        raise TypeError(
            f"Command '{command_name}' must be a ReachTargetCommand, got {type(command)}"
        )
    pos_error: torch.Tensor = command.metrics["pos_error"]
    return torch.exp(-(pos_error / std)**2)

def quat_reach_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    quat_std: float,
    pos_std: float,
) -> torch.Tensor:
    """
    Gaussian reward for reaching target orientation.
    
    Args:
        env: The environment
        command_name: Name of the command term
        quat_std: Standard deviation for the Gaussian distribution for orientation
        pos_std: Standard deviation for the Gaussian distribution for position

    Returns:
        Tensor of shape (num_envs,): reward value (positive for maximization)
    """
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, ReachTargetCommand):
        raise TypeError(
            f"Command '{command_name}' must be a ReachTargetCommand, got {type(command)}"
        )
    quat_error: torch.Tensor = command.metrics["quat_error"]
    pos_error: torch.Tensor = command.metrics["pos_error"]
    pos_error_reward = torch.exp(-(pos_error / pos_std)**2)
    quat_error_reward = pos_error_reward * torch.exp(-(quat_error / quat_std)**2)
    return quat_error_reward


def align_stage_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    quat_std: float,
    pos_std: float,
) -> torch.Tensor:
    """
    Reward for align stage (to peg peak).
    Only active when stage == 0.
    """
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, ReachTargetCommand):
        raise TypeError(
            f"Command '{command_name}' must be a ReachTargetCommand, got {type(command)}"
        )
    stage = command.stage
    align_pos = command.metrics["align_pos_error"]
    align_quat = command.metrics["align_quat_error"]

    pos_reward = torch.exp(-(align_pos / pos_std) ** 2)
    quat_reward = torch.exp(-(align_quat / quat_std) ** 2)

    reward = pos_reward * quat_reward
    # gate by being in align stage (stage == 0)
    gate = (stage == 0).float()
    return reward * gate


def insert_stage_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    quat_std: float,
    pos_std: float,
) -> torch.Tensor:
    """Reward for insert stage (to peg root). Only active when stage == 1."""
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, ReachTargetCommand):
        raise TypeError(
            f"Command '{command_name}' must be a ReachTargetCommand, got {type(command)}"
        )
    stage = command.stage
    insert_pos = command.metrics["insert_pos_error"]
    insert_quat = command.metrics["insert_quat_error"]

    pos_reward = torch.exp(-(insert_pos / pos_std) ** 2)
    quat_reward = torch.exp(-(insert_quat / quat_std) ** 2)

    reward = pos_reward * quat_reward
    gate = (stage == 1).float()
    return reward * gate