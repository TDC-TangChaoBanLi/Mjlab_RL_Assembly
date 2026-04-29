




from re import S

import torch


from mjlab.envs import ManagerBasedRlEnv



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

