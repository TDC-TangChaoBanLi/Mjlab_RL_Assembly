

import torch


from mjlab.envs import ManagerBasedRlEnv



from .commands import ReachTargetCommand



def pos_reach_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Gaussian reward for reaching target position."""
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
    """Gaussian reward for reaching target orientation."""
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
    """Reward for align stage (to peg peak). Only active when stage == 0."""
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
