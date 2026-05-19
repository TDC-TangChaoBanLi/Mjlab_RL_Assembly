from typing import Literal
from dataclasses import dataclass, field

import torch


from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTermCfg, CommandTerm
from mjlab.utils.lab_api.math import sample_uniform


from mjlab_rl_assembly.cfg.constants import (
    EE_SITE_NAME,
    PEG_SITE_NAME,
    PEG_PEAK_SITE_NAME,
    UR5E_ENTITY_NAME,
    PEG_ENTITY_NAME,
    FORCE_SENSOR_NAME,
    TORQUE_SENSOR_NAME,
)

from mjlab.sensor import BuiltinSensor



# command term config
@dataclass(kw_only=True)
class ReachTargetCommandCfg(CommandTermCfg):
    """
    Configuration for reaching a virtual target position.
    Args:
        align_pos_tolerance: Position tolerance for aligning to the peak of the peg.
        align_quat_tolerance: Quaternion tolerance for aligning to the peak of the peg.
        insert_pos_tolerance: Position tolerance for inserting the peg.
        insert_quat_tolerance: Quaternion tolerance for inserting the peg.
        failure_pos_tolerance: Position tolerance for failure.
        failure_pose_cnt_tol: Number of consecutive failures required to declare a failure.
        failure_ft_cnt_tol: Number of consecutive failures required to declare a failure.
        force_limit: Maximum force limit.
        torque_limit: Maximum torque limit.
        viz: Visualization configuration.
    Returns:
        ReachTargetCommand: The command term.
       """
    align_pos_tolerance: float = 0.01
    align_quat_tolerance: float = 0.05
    insert_pos_tolerance: float = 0.005
    insert_quat_tolerance: float = 0.01
    failure_pos_tolerance: float = 0.2
    failure_pose_cnt_tol: int = 10  # 连续多少次超出容差才判断为失败
    failure_ft_cnt_tol: int = 10  # 连续多少次超出容差才判断为失败
    force_limit: float = 60.0
    torque_limit: float = 30.0
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
        self.ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.ee_quat[:, 0] = 1.0  # 初始化为单位四元数
        # stage: 0=align (peak), 1=insert (root)
        self.stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # store both peak and root target poses (world frame)
        self.peak_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.peak_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.peak_quat[:, 0] = 1.0

        self.root_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.root_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.root_quat[:, 0] = 1.0

        # self.episode_success = torch.zeros(self.num_envs, device=self.device)

        # metrics
        self.metrics["pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["quat_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stage"] = self.stage.clone().float()

        # align stage metrics (to peak)
        self.metrics["align_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["align_quat_error"] = torch.zeros(self.num_envs, device=self.device)
        # insert stage metrics (to root)
        self.metrics["insert_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["insert_quat_error"] = torch.zeros(self.num_envs, device=self.device)


        # 连续位置失败
        self.failure_pose_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.metrics["failure_pose"] = torch.zeros(self.num_envs, device=self.device)


        # 连续力矩失败
        self.failure_ft_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.metrics["failure_ft"] = torch.zeros(self.num_envs, device=self.device)
        
        # 力矩传感器值
        self.metrics["ft_sensor"] = torch.zeros(self.num_envs, 6, device=self.device)
        self._ft_filter_state = torch.zeros(self.num_envs, 6, device=self.device)

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
            self.ee_pos = robot.data.site_pos_w[:, ee_site_id] # End-effector position in world frame
            self.ee_quat = robot.data.site_quat_w[:, ee_site_id] # End-effector quaternion in world frame
        else:
            self.ee_pos = robot.data.site_pos_w[:, 0]  # Default to first site
            self.ee_quat = robot.data.site_quat_w[:, 0]  # Default to first site
        
        # Calculate position error and quaternion error
        pos_error = torch.norm(self.target_pos - self.ee_pos, dim=-1) # L2 norm

        dot = torch.sum(self.target_quat * self.ee_quat, dim=-1)
        dot = torch.clamp(torch.abs(dot), max=1.0)
        quat_error = 2.0 * torch.acos(dot) # err = 2 * arccos(|q_t \dot q_e|) 内积计算四元数角度

        # align errors (to peak)
        align_pos_error = torch.norm(self.peak_pos - self.ee_pos, dim=-1)
        dot_align = torch.sum(self.peak_quat * self.ee_quat, dim=-1)
        dot_align = torch.clamp(torch.abs(dot_align), max=1.0)
        align_quat_error = 2.0 * torch.acos(dot_align)

        # insert errors (to root)
        insert_pos_error = torch.norm(self.root_pos - self.ee_pos, dim=-1)
        dot_insert = torch.sum(self.root_quat * self.ee_quat, dim=-1)
        dot_insert = torch.clamp(torch.abs(dot_insert), max=1.0)
        insert_quat_error = 2.0 * torch.acos(dot_insert)


        ############### 连续位置失败 #############
        # failure is determined by consecutive pos_error > failure_pos_tolerance
        pos_error_exceeds = (pos_error > self.cfg.failure_pos_tolerance)
        # 更新连续位置失败计数器
        self.failure_pose_counter[pos_error_exceeds] += 1
        self.failure_pose_counter[~pos_error_exceeds] = 0  # 重置计数器
        # 只有连续超出次数达到阈值才判断为失败
        failure_pose = (self.failure_pose_counter >= self.cfg.failure_pose_cnt_tol).float()

        ############### 连续力矩失败 #############
        # failure is determined by consecutive ft_exceed_limit
        ft_exceed_limit = (torch.norm(self._ft_filter_state, dim=-1) > self.cfg.force_limit) | (torch.norm(self._ft_filter_state, dim=-1) > self.cfg.torque_limit)
        # 更新连续力矩失败计数器
        self.failure_ft_counter[ft_exceed_limit] += 1
        self.failure_ft_counter[~ft_exceed_limit] = 0  # 重置计数器
        # 只有连续超出次数达到阈值才判断为失败
        failure_ft = (self.failure_ft_counter >= self.cfg.failure_ft_cnt_tol).float()

        # determine success depending on stage
        at_goal_align = torch.logical_and(
            align_pos_error.abs() < self.cfg.align_pos_tolerance,
            align_quat_error.abs() < self.cfg.align_quat_tolerance,
        )

        at_goal_insert = torch.logical_and(
            insert_pos_error.abs() < self.cfg.insert_pos_tolerance,
            insert_quat_error.abs() < self.cfg.insert_quat_tolerance,
        )

        # update stage: once aligned, move to insert (latched)
        stage_long = self.stage.clone()
        stage_long[at_goal_align] = 1
        self.stage = stage_long

        # success is determined by insert success (stage==1 and at_goal_insert)
        success = (self.stage == 1).float() * at_goal_insert.float()

        # update metrics
        self.metrics["pos_error"] = pos_error
        self.metrics["quat_error"] = quat_error
        self.metrics["success"] = success
        self.metrics["stage"] = self.stage.clone().float()
        self.metrics["failure_pose"] = failure_pose
        self.metrics["failure_ft"] = failure_ft


        self.metrics["align_pos_error"] = align_pos_error
        self.metrics["align_quat_error"] = align_quat_error
        self.metrics["insert_pos_error"] = insert_pos_error
        self.metrics["insert_quat_error"] = insert_quat_error

        # Update force/torque sensor metrics with EWMA filtering
        self._update_ft_metrics()

    def _update_ft_metrics(self) -> None:
        """
        Update force/torque sensor metrics with EWMA filtering.
        """
        # Get force/torque sensors
        force_sensor: BuiltinSensor = self._env.scene[UR5E_ENTITY_NAME + "/" + FORCE_SENSOR_NAME]
        torque_sensor: BuiltinSensor = self._env.scene[UR5E_ENTITY_NAME + "/" + TORQUE_SENSOR_NAME]

        raw_force = force_sensor.data  # (num_envs, 3)
        raw_torque = torque_sensor.data  # (num_envs, 3)

        # Combine force and torque
        raw = torch.cat([raw_force, raw_torque], dim=-1)

        # Check for NaN or Inf values
        has_invalid = torch.any(torch.isnan(raw) | torch.isinf(raw), dim=-1)
        
        # Replace invalid rows with previous filtered state
        raw = torch.where(
            has_invalid.unsqueeze(-1),
            self._ft_filter_state,
            raw
        )

        # EWMA filtering: y_t = alpha * x_t + (1 - alpha) * y_{t-1}
        alpha = 0.01
        self._ft_filter_state = alpha * raw + (1.0 - alpha) * self._ft_filter_state

        # Update metrics
        self.metrics["ft_sensor"] = self._ft_filter_state

    def compute_success(self) -> torch.Tensor:
        # success based on insert stage
        return self.metrics["success"]

    def _resample_command(self, env_ids: torch.Tensor) -> None: # 继承
        """
        重新生成目标位置和 peg 位置
        """
        self.failure_pose_counter[env_ids] = 0
        self.failure_ft_counter[env_ids] = 0
        self.stage[env_ids] = 0
        self._ft_filter_state[env_ids] = torch.zeros(len(env_ids), 6, device=self.device)
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
        # Read both peak and root sites from peg
        # peak
        peak_site_ids, peak_site_names = peg_entity.find_sites(PEG_PEAK_SITE_NAME)
        if len(peak_site_ids) > 0 and peak_site_names[0] == PEG_PEAK_SITE_NAME:
            peak_id = peak_site_ids[0]
            peak_pos = peg_entity.data.site_pos_w[:, peak_id]
            peak_quat = peg_entity.data.site_quat_w[:, peak_id]
        else:
            peak_pos = peg_entity.data.site_pos_w[:, 0]
            peak_quat = peg_entity.data.site_quat_w[:, 0]

        # root (peg site)
        root_site_ids, root_site_names = peg_entity.find_sites(PEG_SITE_NAME)
        if len(root_site_ids) > 0 and root_site_names[0] == PEG_SITE_NAME:
            root_id = root_site_ids[0]
            root_pos = peg_entity.data.site_pos_w[:, root_id]
            root_quat = peg_entity.data.site_quat_w[:, root_id]
        else:
            root_pos = peg_entity.data.site_pos_w[:, 0]
            root_quat = peg_entity.data.site_quat_w[:, 0]

        # write into command buffers
        self.peak_pos[:] = peak_pos
        self.peak_quat[:] = peak_quat
        self.root_pos[:] = root_pos
        self.root_quat[:] = root_quat

        # set active target depending on stage per-env
        # stage == 0 -> align -> target is peak
        # stage == 1 -> insert -> target is root
        if self.num_envs == 1:
            if int(self.stage.item()) == 0:
                self.target_pos[:] = self.peak_pos
                self.target_quat[:] = self.peak_quat
            else:
                self.target_pos[:] = self.root_pos
                self.target_quat[:] = self.root_quat
        else:
            mask_align = (self.stage == 0)
            mask_insert = (self.stage == 1)
            if mask_align.any():
                self.target_pos[mask_align] = self.peak_pos[mask_align]
                self.target_quat[mask_align] = self.peak_quat[mask_align]
            if mask_insert.any():
                self.target_pos[mask_insert] = self.root_pos[mask_insert]
                self.target_quat[mask_insert] = self.root_quat[mask_insert]
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