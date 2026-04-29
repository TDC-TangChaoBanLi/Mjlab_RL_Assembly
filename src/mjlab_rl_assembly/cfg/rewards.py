

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
