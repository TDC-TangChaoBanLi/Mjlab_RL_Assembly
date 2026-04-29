from typing import Literal
from dataclasses import dataclass, field

import torch


from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTermCfg, CommandTerm
from mjlab.utils.lab_api.math import sample_uniform


from mjlab_rl_assembly.cfg.constants import EE_SITE_NAME, PEG_SITE_NAME, UR5E_ENTITY_NAME, PEG_ENTITY_NAME



# command term config
@dataclass(kw_only=True)
class ReachTargetCommandCfg(CommandTermCfg):
    """Configuration for reaching a virtual target position."""
    pos_tolerance: float = 0.005
    quat_tolerance: float = 0.01
    # difficulty: Literal["fixed", "dynamic"] = "fixed"

    # @dataclass
    # class TargetPositionRangeCfg:
    #     """Configuration for target position sampling in dynamic mode."""
    #     x: tuple[float, float] = (0.3, 0.5)
    #     y: tuple[float, float] = (-0.2, 0.2)
    #     z: tuple[float, float] = (0.2, 0.4)

    # # Only used in dynamic mode.
    # target_position_range: TargetPositionRangeCfg = field(
    #     default_factory=TargetPositionRangeCfg
    # )

    @dataclass
    class VizCfg:
        target_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.3)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env):
        return ReachTargetCommand(self, env)

# command term
class ReachTargetCommand(CommandTerm):
    """Command for reaching a virtual target position with end-effector."""
    cfg: ReachTargetCommandCfg

    def __init__(self, cfg: ReachTargetCommandCfg, env):
        super().__init__(cfg, env)
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.target_quat[:, 0] = 1.0  # 初始化为单位四元数
        # self.episode_success = torch.zeros(self.num_envs, device=self.device)

        self.metrics["pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["quat_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)
        # self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target_pos

    def _update_metrics(self) -> None: # 继承
        """
        更新内部指标参数
        """
        # Get end-effector position
        robot: Entity = self._env.scene[UR5E_ENTITY_NAME]

        # Get end-effector position in world frame
        (site_ids, site_names) = robot.find_sites(EE_SITE_NAME)
        if site_names[0]  == EE_SITE_NAME:
            ee_site_id = site_ids[0]
            ee_pos_w = robot.data.site_pos_w[:, ee_site_id] # End-effector position in world frame
            ee_quat_w = robot.data.site_quat_w[:, ee_site_id] # End-effector quaternion in world frame
        else:
            ee_pos_w = robot.data.site_pos_w[:, 0]  # Default to first site
            ee_quat_w = robot.data.site_quat_w[:, 0]  # Default to first site
        
        # Calculate position error and quaternion error
        pos_error = torch.norm(self.target_pos - ee_pos_w, dim=-1) # L2 norm

        dot = torch.sum(self.target_quat * ee_quat_w, dim=-1)
        dot = torch.clamp(torch.abs(dot), max=1.0)
        quat_error = 2.0 * torch.acos(dot) # err = 2 * arccos(|q_t \dot q_e|) 内积计算四元数角度
        
        success = torch.logical_and(pos_error.abs() < self.cfg.pos_tolerance, 
                                    quat_error.abs() < self.cfg.quat_tolerance).float()


        # Latch episode_success to 1 once goal is reached
        # self.episode_success = torch.maximum(self.episode_success, at_goal)

        self.metrics["pos_error"] = pos_error
        self.metrics["quat_error"] = quat_error
        self.metrics["success"] = success
        # self.metrics["episode_success"] = self.episode_success

    def compute_success(self) -> torch.Tensor:
        return self.metrics["at_goal"]

    def _resample_command(self, env_ids: torch.Tensor) -> None: # 继承
        """
        重新生成目标位置和 peg 位置
        """
        pass
        # n = len(env_ids)

        # # Reset episode success for resampled envs
        # self.episode_success[env_ids] = 0.0

        # # Get peg entity
        # peg_entity: Entity = self._env.scene[PEG_ENTITY_NAME]
        
        # # Randomize peg position
        # if self.cfg.difficulty == "fixed":
        #     # Fixed peg position
        #     sample_pos = torch.tensor(
        #         [0.5, 0.0, 0.5], device=self.device, dtype=torch.float32
        #     ).expand(n, 3)
        # else:
        #     # Dynamic peg position - randomize within range
        #     assert self.cfg.difficulty == "dynamic"
        #     r = self.cfg.target_position_range
        #     lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
        #     upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
        #     sample_pos = sample_uniform(lower, upper, (n, 3), device=self.device)
        
        # # Add env origins
        # peg_pos = torch.zeros(n, 3, device=self.device)
        # peg_pos = sample_pos + self._env.scene.env_origins[env_ids] # position
        
        # # Set peg position
        # peg_quat = torch.zeros(n, 4, device=self.device) # orientation
        # peg_quat[:, 0] = 1.0  # Identity quaternion 
        
        # # Write peg pose to simulation
        # peg_pose = torch.cat([peg_pos, peg_quat], dim=-1) # position and orientation
        # peg_entity.write_mocap_pose_to_sim(peg_pose, env_ids)



    def _update_command(self) -> None: # 继承
        # Get peg entity
        peg_entity: Entity = self._env.scene[PEG_ENTITY_NAME]
        # Set target position to peg's PEG_SITE_NAME site with z-axis offset
        # First, find the site ID for PEG_SITE_NAME
        (site_ids, site_names) = peg_entity.find_sites(PEG_SITE_NAME)
        if site_names[0] == PEG_SITE_NAME:
            peg_site_id = site_ids[0]
            peg_site_pos = peg_entity.data.site_pos_w[:, peg_site_id]
            peg_site_quat = peg_entity.data.site_quat_w[:, peg_site_id]

            self.target_pos[:] = peg_site_pos
            self.target_quat[:] = peg_site_quat
        else:
            # Fallback to peg root position if site not found
            self.target_pos[:] = peg_entity.data.site_pos_w[:, 0]
            self.target_quat[:] = peg_entity.data.site_quat_w[:, 0]
        # self.target_pos[env_ids] = peg_pos
        # self.target_quat[env_ids] = peg_quat

    def _debug_vis_impl(self, visualizer) -> None: # 继承
        """
        创建可视化球体表示目标位置
        """
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        for batch in env_indices:
            target_pos = self.target_pos[batch].cpu().numpy()
            visualizer.add_sphere(
                center=target_pos,
                radius=0.03,
                color=self.cfg.viz.target_color,
                label=f"target_position_{batch}",
            )
