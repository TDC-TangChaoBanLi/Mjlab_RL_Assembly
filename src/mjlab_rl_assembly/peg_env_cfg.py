

from __future__ import annotations

# stdlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal
# third party
import mujoco
from warp import init

# mjlab
from mjlab import sim
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg


from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise


from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg, CommandTerm
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg # 奖励函数
from mjlab.managers.termination_manager import TerminationTermCfg # 终止条件
from mjlab.managers.curriculum_manager import CurriculumTermCfg # 课程管理
from mjlab.envs import ManagerBasedRlEnv

from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mjlab.sim import MujocoCfg, SimulationCfg

from mjlab.envs import mdp
from mjlab.tasks.manipulation import mdp as manipulation_mdp

import numpy as np
import torch
import mink
from mjlab.utils.lab_api.math import sample_uniform, quat_apply, quat_from_euler_xyz


from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


import math



# import jax
# import jax.numpy as jnp
# import mujoco as mj
# import mujoco.mjx as mjx
# from jaxlie import SE3

# from mjinx.problem import Problem
# from mjinx.components.tasks import FrameTask
# from mjinx.components.barriers import JointBarrier
# from mjinx.solvers import LocalIKSolver
# from mjinx.configuration import integrate

############### Entity ###############


# mjcf 文件路径
_UE5E_MJCF: Path = Path(__file__).parent / "mjcf"/"UR5e.xml"

# 机器人关节名称
_UE5E_ACTUATOR_JOINTS = (
    "ur_shoulder_pan_joint",
    "ur_shoulder_lift_joint",
    "ur_elbow_joint",
    "ur_wrist_1_joint",
    "ur_wrist_2_joint",
    "ur_wrist_3_joint",
    # "robotiq_85_left_knuckle_joint"
    )

# 初始化关节状态
_UE5E_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    rot=(0.0, 0.0, 0.0, 1.0),
    joint_pos={
        _UE5E_ACTUATOR_JOINTS[0]: 0.0, 
        _UE5E_ACTUATOR_JOINTS[1]: -1.57, 
        _UE5E_ACTUATOR_JOINTS[2]: 0.0, 
        _UE5E_ACTUATOR_JOINTS[3]: -1.57, 
        _UE5E_ACTUATOR_JOINTS[4]: 0.0, 
        _UE5E_ACTUATOR_JOINTS[5]: 0.0, 
        # _UE5E_ACTUATOR_JOINTS[6]: 0.0
        },
    joint_vel={".*": 0.0},
)

def _get_ur5e_spec() -> mujoco.MjSpec:
    """
    从 mjcf 文件路径加载 mujoco 模型。
    """
    return mujoco.MjSpec.from_file(str(_UE5E_MJCF))

def _get_ur5e_entity_cfg() -> EntityCfg:
    """
    创建机械臂实体配置。
    """
    return EntityCfg(
        spec_fn=_get_ur5e_spec,
        articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=_UE5E_ACTUATOR_JOINTS),)),
        init_state=_UE5E_INIT_STATE,
    )



# peg 实体配置
_PEG_MJCF: Path = Path(__file__).parent / "mjcf"/"peg.xml"
def _get_peg_spec() -> mujoco.MjSpec:
    """
    从 mjcf 文件路径加载 mujoco 模型。
    """
    spec = mujoco.MjSpec.from_file(str(_PEG_MJCF))
    # Set the peg body as mocap to allow pose control
    spec.worldbody.first_body().mocap = True
    return spec

def _get_peg_entity_cfg() -> EntityCfg:
    """
    创建 peg 实体配置。
    """
    return EntityCfg(
        spec_fn=_get_peg_spec,
        init_state = EntityCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={".*": 0.0},
        ),
    )


# scene 配置
scene = SceneCfg(
        num_envs=1,
        env_spacing=1.0,
        terrain=TerrainEntityCfg(terrain_type="plane"), # 地形
        entities={
            "UR5e": _get_ur5e_entity_cfg(), # 机械臂
            "peg": _get_peg_entity_cfg()  # 机械臂端
            }, 
        # sensors=(ee_ground_collision_cfg,),
    )


############### Observations ###############
_EE_SITE_NAME = "_hole"
_PEG_SITE_NAME = "_peg"

# command term config
@dataclass(kw_only=True)
class ReachTargetCommandCfg(CommandTermCfg):
    """Configuration for reaching a virtual target position."""
    pos_tolerance: float = 0.05
    quat_tolerance: float = 0.1
    difficulty: Literal["fixed", "dynamic"] = "fixed"

    @dataclass
    class TargetPositionRangeCfg:
        """Configuration for target position sampling in dynamic mode."""
        x: tuple[float, float] = (0.3, 0.5)
        y: tuple[float, float] = (-0.2, 0.2)
        z: tuple[float, float] = (0.2, 0.4)

    # Only used in dynamic mode.
    target_position_range: TargetPositionRangeCfg = field(
        default_factory=TargetPositionRangeCfg
    )

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
        self.episode_success = torch.zeros(self.num_envs, device=self.device)

        self.metrics["pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["quat_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target_pos

    def _update_metrics(self) -> None: # 继承
        """
        更新内部指标参数
        """
        # Get end-effector position
        robot: Entity = self._env.scene["UR5e"]

        # Get end-effector position in world frame
        (site_ids, site_names) = robot.find_sites(_EE_SITE_NAME)
        if site_names[0]  == _EE_SITE_NAME:
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
        
        at_goal = (torch.logical_and(pos_error < self.cfg.pos_tolerance, 
                                    quat_error < self.cfg.quat_tolerance)).float()


        # Latch episode_success to 1 once goal is reached
        self.episode_success = torch.maximum(self.episode_success, at_goal)

        self.metrics["pos_error"] = pos_error
        self.metrics["quat_error"] = quat_error
        self.metrics["at_goal"] = at_goal
        self.metrics["episode_success"] = self.episode_success

    def compute_success(self) -> torch.Tensor:
        return self.metrics["at_goal"]

    def _resample_command(self, env_ids: torch.Tensor) -> None: # 继承
        """
        重新生成目标位置和 peg 位置
        """
        n = len(env_ids)

        # Reset episode success for resampled envs
        self.episode_success[env_ids] = 0.0

        # Get peg entity
        peg_entity: Entity = self._env.scene["peg"]
        
        # Randomize peg position
        if self.cfg.difficulty == "fixed":
            # Fixed peg position
            sample_pos = torch.tensor(
                [0.5, 0.0, 0.5], device=self.device, dtype=torch.float32
            ).expand(n, 3)
        else:
            # Dynamic peg position - randomize within range
            assert self.cfg.difficulty == "dynamic"
            r = self.cfg.target_position_range
            lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
            upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
            sample_pos = sample_uniform(lower, upper, (n, 3), device=self.device)
        
        # Add env origins
        peg_pos = torch.zeros(n, 3, device=self.device)
        peg_pos = sample_pos + self._env.scene.env_origins[env_ids] # position
        
        # Set peg position
        peg_quat = torch.zeros(n, 4, device=self.device) # orientation
        peg_quat[:, 0] = 1.0  # Identity quaternion 
        
        # Write peg pose to simulation
        peg_pose = torch.cat([peg_pos, peg_quat], dim=-1) # position and orientation
        # peg_entity.write_mocap_pose_to_sim(peg_pose, env_ids)



    def _update_command(self) -> None: # 继承
        # Get peg entity
        peg_entity: Entity = self._env.scene["peg"]
        # Set target position to peg's _PEG_SITE_NAME site with z-axis offset
        # First, find the site ID for _PEG_SITE_NAME
        (site_ids, site_names) = peg_entity.find_sites(_PEG_SITE_NAME)
        if site_names[0] == _PEG_SITE_NAME:
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


def _find_site_name_in_model(mj_model, candidates):
    for name in candidates:
        site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return name
    raise ValueError(f"Cannot find site in MuJoCo model. Tried: {candidates}")


def _find_mocap_body_for_entity(mj_model, entity_name: str):
    for body_id in range(mj_model.nbody):
        mocap_id = int(mj_model.body_mocapid[body_id])
        if mocap_id < 0:
            continue

        body_name = mujoco.mj_id2name(
            mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        if body_name is None:
            continue

        if body_name == entity_name or body_name.startswith(entity_name + "/"):
            return body_id, mocap_id

    return None, None


def _set_entity_mocap_pose_in_data(
    mj_model,
    mj_data,
    entity_name: str,
    pos_local_np: np.ndarray,
    quat_wxyz_np: np.ndarray,
) -> bool:
    _, mocap_id = _find_mocap_body_for_entity(mj_model, entity_name)

    if mocap_id is None:
        return False

    mj_data.mocap_pos[mocap_id] = pos_local_np
    mj_data.mocap_quat[mocap_id] = quat_wxyz_np
    return True


def _infer_robot_joint_names_from_mj_model(mj_model, robot_prefix: str):
    joint_names = []

    for jid in range(mj_model.njnt):
        jtype = int(mj_model.jnt_type[jid])

        if jtype not in [
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ]:
            continue

        name = mujoco.mj_id2name(
            mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            jid,
        )

        if name is None:
            continue

        if name.startswith(robot_prefix + "/"):
            joint_names.append(name)

    if len(joint_names) == 0:
        raise ValueError(
            f"Cannot infer robot joints with prefix '{robot_prefix}/'. "
            "Please check joint names in MJCF."
        )

    return joint_names


def _get_joint_qpos_indices(mj_model, joint_names):
    qpos_indices = []

    for joint_name in joint_names:
        jid = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if jid < 0:
            raise ValueError(f"Joint '{joint_name}' not found in MuJoCo model.")

        jtype = int(mj_model.jnt_type[jid])

        if jtype not in [
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ]:
            raise ValueError(
                f"Joint '{joint_name}' is not hinge/slide joint. "
                "This helper only supports 1-DoF robot joints."
            )

        qpos_indices.append(int(mj_model.jnt_qposadr[jid]))

    return np.asarray(qpos_indices, dtype=np.int64)


def _build_full_qpos_for_mink(
    mj_model,
    robot_joint_pos_np,
    robot_joint_qpos_indices,
):
    qpos_full = mj_model.qpos0.copy()

    if len(robot_joint_pos_np) != len(robot_joint_qpos_indices):
        raise ValueError(
            f"robot_joint_pos length = {len(robot_joint_pos_np)}, "
            f"but robot_joint_qpos_indices length = {len(robot_joint_qpos_indices)}."
        )

    qpos_full[robot_joint_qpos_indices] = robot_joint_pos_np
    return qpos_full


def _make_mink_configuration(mj_model, qpos_full):
    try:
        configuration = mink.Configuration(mj_model)
        configuration.update(qpos_full)
    except TypeError:
        configuration = mink.Configuration(mj_model, qpos_full)

    return configuration


def _geom_exists(mj_model, geom_name: str) -> bool:
    geom_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_name,
    )
    return geom_id >= 0


def _filter_existing_geoms(mj_model, geom_names, group_name: str):
    valid = []
    missing = []

    for name in geom_names:
        if _geom_exists(mj_model, name):
            valid.append(name)
        else:
            missing.append(name)

    if len(missing) > 0:
        print(
            f"[reset IK warning] Some geoms in '{group_name}' do not exist "
            f"and will be ignored: {missing}"
        )

    return valid


def _get_geom_name(mj_model, geom_id: int) -> str:
    name = mujoco.mj_id2name(
        mj_model,
        mujoco.mjtObj.mjOBJ_GEOM,
        int(geom_id),
    )
    return "" if name is None else name


def _has_robot_self_collision(
    mj_model,
    mj_data,
    robot_geom_names,
    min_penetration: float = 1e-5,
) -> bool:
    robot_geom_set = set(robot_geom_names)

    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]

        g1 = _get_geom_name(mj_model, contact.geom1)
        g2 = _get_geom_name(mj_model, contact.geom2)

        if g1 in robot_geom_set and g2 in robot_geom_set:
            if contact.dist < -float(min_penetration):
                return True

    return False


def _range_min_max(value):
    if isinstance(value, (tuple, list)):
        a = float(value[0])
        b = float(value[1])
    else:
        a = float(value)
        b = float(value)

    return min(a, b), max(a, b)


def _sample_peg_root_pose_ring_bucket(
    n: int,
    device,
    pose_range: dict,
):
    inner_radius = float(pose_range.get("inner_radius", 0.1))
    outer_radius = float(pose_range.get("outer_radius", 0.5))

    if inner_radius < 0.0:
        raise ValueError("inner_radius must be non-negative.")

    if outer_radius <= inner_radius:
        raise ValueError("outer_radius must be larger than inner_radius.")

    z_min, z_max = _range_min_max(
        (
            pose_range.get("z_start", 0.3),
            pose_range.get("z_end", 0.7),
        )
    )

    roll_min, roll_max = _range_min_max(
        pose_range.get("roll_range", (-0.2, 0.2))
    )
    pitch_min, pitch_max = _range_min_max(
        pose_range.get("pitch_range", (-0.2, 0.2))
    )
    yaw_offset_min, yaw_offset_max = _range_min_max(
        pose_range.get("yaw_offset_range", (-0.5, 0.5))
    )

    u = torch.rand(n, device=device, dtype=torch.float32)
    v = torch.rand(n, device=device, dtype=torch.float32)

    r = torch.sqrt(
        inner_radius**2 + u * (outer_radius**2 - inner_radius**2)
    )
    theta = 2.0 * math.pi * v

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    z = z_min + (z_max - z_min) * torch.rand(
        n,
        device=device,
        dtype=torch.float32,
    )

    roll = roll_min + (roll_max - roll_min) * torch.rand(
        n,
        device=device,
        dtype=torch.float32,
    )
    pitch = pitch_min + (pitch_max - pitch_min) * torch.rand(
        n,
        device=device,
        dtype=torch.float32,
    )

    yaw_base = torch.atan2(y, x) + math.pi / 2.0
    yaw_offset = yaw_offset_min + (yaw_offset_max - yaw_offset_min) * torch.rand(
        n,
        device=device,
        dtype=torch.float32,
    )
    yaw = yaw_base + yaw_offset

    peg_pos_local = torch.stack([x, y, z], dim=-1)
    peg_quat_wxyz = quat_from_euler_xyz(roll, pitch, yaw)

    return peg_pos_local, peg_quat_wxyz


def _compute_target_pose_from_peg_target(
    mj_model,
    peg_site_name: str,
    peg_entity_name: str,
    peg_pos_local_np: np.ndarray,
    peg_quat_wxyz_np: np.ndarray,
    z_offset: float,
):
    tmp_data = mujoco.MjData(mj_model)

    ok = _set_entity_mocap_pose_in_data(
        mj_model=mj_model,
        mj_data=tmp_data,
        entity_name=peg_entity_name,
        pos_local_np=peg_pos_local_np,
        quat_wxyz_np=peg_quat_wxyz_np,
    )

    if not ok:
        raise RuntimeError(
            f"Cannot find mocap body for entity '{peg_entity_name}'. "
            "Please check whether peg is controlled by mocap."
        )

    mujoco.mj_forward(mj_model, tmp_data)

    peg_site_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_SITE,
        peg_site_name,
    )

    if peg_site_id < 0:
        raise ValueError(f"Peg site '{peg_site_name}' not found in MuJoCo model.")

    site_pos = tmp_data.site_xpos[peg_site_id].copy()
    site_xmat = tmp_data.site_xmat[peg_site_id].reshape(3, 3).copy()

    site_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(site_quat, site_xmat.reshape(-1))

    offset_local = np.array([0.0, 0.0, float(z_offset)], dtype=np.float64)

    target_pos = site_pos + site_xmat @ offset_local
    target_quat = site_quat

    return target_pos, target_quat


def reset_peg_pose_and_ur5e_ik(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("UR5e"),
    z_offset: float = -0.1,
    ik_iterations: int = 80,
    ik_dt: float = 0.01,
    ik_position_cost: float = 1.0,
    ik_orientation_cost: float = 0.1,
    ik_solver: str = "daqp",
    ik_damping: float = 1e-4,
    ik_pos_tol: float = 5e-3,
    ik_ori_tol: float = 5e-2,
    max_resample_attempts: int = 50,
    self_collision_min_distance: float = 0.02,
    self_collision_detection_distance: float = 0.15,
    self_collision_min_penetration: float = 1e-5,
    safe_peg_pos: tuple[float, float, float] = (0.0, 0.0, -10.0),
) -> int:
    """
    Reset peg pose randomly and solve UR5e IK so the EE site tracks the peg site.

    流程：
    1. 将 peg entity 设置到安全位置；
    2. 对每个 env 采样随机目标位姿；
    3. 计算目标点处的 IK；
    4. 检查机械臂自身是否碰撞；
    5. 若 IK 失败或自碰撞，则重新采样该 env 的目标位姿并重新 IK；
    6. 成功后写入机械臂关节角，并将 peg entity 设置到对应目标位置；
    7. 若超过 max_resample_attempts 仍失败，则返回 1。

    返回：
        0: 所有 env 初始化成功；
        1: 至少一个 env 超过最大重采样次数后仍失败。
    """

    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            device=env.device,
            dtype=torch.long,
        )
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    if len(env_ids) == 0:
        return 0

    peg_entity: Entity = env.scene[asset_cfg.name]
    robot_entity: Entity = env.scene[robot_cfg.name]
    mj_model = env.sim.mj_model

    # ----------------------------------------------------
    # 0. 准备 site、joint、geom 信息
    # ----------------------------------------------------
    ee_site_name = _find_site_name_in_model(
        mj_model,
        candidates=[
            f"{robot_cfg.name}/{_EE_SITE_NAME}",
            _EE_SITE_NAME,
        ],
    )

    peg_site_name = _find_site_name_in_model(
        mj_model,
        candidates=[
            f"{asset_cfg.name}/{_PEG_SITE_NAME}",
            _PEG_SITE_NAME,
        ],
    )

    robot_joint_names = _infer_robot_joint_names_from_mj_model(
        mj_model,
        robot_prefix=robot_cfg.name,
    )

    robot_joint_qpos_indices = _get_joint_qpos_indices(
        mj_model,
        robot_joint_names,
    )

    num_robot_joints = len(robot_joint_qpos_indices)

    robot_self_collision_geom_names = [
        "UR5e/COLLISION_ur_base_link_inertia_0",
        "UR5e/COLLISION_ur_shoulder_link_0",
        "UR5e/COLLISION_ur_upper_arm_link_0",
        "UR5e/COLLISION_ur_forearm_link_0",
        "UR5e/COLLISION_ur_wrist_1_link_0",
        "UR5e/COLLISION_ur_wrist_2_link_0",
        "UR5e/COLLISION_ur_wrist_3_link_0",
        "UR5e/COLLISION_lens_link_0",
        "UR5e/COLLISION_lens_link_1",
        "UR5e/COLLISION_lens_link_2",
        "UR5e/COLLISION_lens_link_3",
        "UR5e/COLLISION_lens_link_4",
        "UR5e/COLLISION_lens_link_5",
        "UR5e/COLLISION_lens_link_6",
        "UR5e/COLLISION_lens_link_7",
        "UR5e/COLLISION_lens_link_8",
    ]

    robot_self_collision_geom_names = _filter_existing_geoms(
        mj_model,
        robot_self_collision_geom_names,
        group_name="robot_self_collision_geom_names",
    )

    limits = [
        mink.ConfigurationLimit(mj_model),
    ]

    if len(robot_self_collision_geom_names) > 0:
        limits.append(
            mink.CollisionAvoidanceLimit(
                model=mj_model,
                geom_pairs=[
                    (
                        robot_self_collision_geom_names,
                        robot_self_collision_geom_names,
                    )
                ],
                gain=0.95,
                minimum_distance_from_collisions=float(self_collision_min_distance),
                collision_detection_distance=float(self_collision_detection_distance),
                bound_relaxation=0.0,
            )
        )
    else:
        print(
            "[reset IK warning] Self-collision avoidance disabled because "
            "robot_self_collision_geom_names is empty."
        )

    # ----------------------------------------------------
    # 1. 先把所有 peg 放到安全位置，避免 peg 干扰 IK
    # ----------------------------------------------------
    all_env_ids = env_ids.clone()
    n_all = len(all_env_ids)

    safe_peg_pos_local_all = torch.tensor(
        safe_peg_pos,
        device=env.device,
        dtype=torch.float32,
    ).unsqueeze(0).expand(n_all, -1)

    safe_peg_pos_w_all = safe_peg_pos_local_all + env.scene.env_origins[all_env_ids]

    safe_peg_quat_wxyz_all = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
        device=env.device,
        dtype=torch.float32,
    ).unsqueeze(0).expand(n_all, -1)

    safe_peg_pose_w_all = torch.cat(
        [safe_peg_pos_w_all, safe_peg_quat_wxyz_all],
        dim=-1,
    )

    peg_entity.write_mocap_pose_to_sim(
        safe_peg_pose_w_all,
        env_ids=all_env_ids,
    )

    default_joint_pos_all = robot_entity.data.default_joint_pos[all_env_ids].clone()
    default_joint_vel_all = robot_entity.data.default_joint_vel[all_env_ids].clone()

    robot_entity.write_joint_state_to_sim(
        default_joint_pos_all,
        default_joint_vel_all,
        env_ids=all_env_ids,
    )

    env.sim.forward()

    # ----------------------------------------------------
    # 2. 重采样循环
    # ----------------------------------------------------
    unresolved_env_ids = all_env_ids.clone()

    accepted_joint_pos: dict[int, np.ndarray] = {}
    accepted_peg_pose_w: dict[int, torch.Tensor] = {}

    for attempt in range(max_resample_attempts):
        if len(unresolved_env_ids) == 0:
            break

        n = len(unresolved_env_ids)

        # 当前这一轮，只给尚未成功的 env 重新采样目标位姿
        target_peg_pos_local, target_peg_quat_wxyz = _sample_peg_root_pose_ring_bucket(
            n=n,
            device=env.device,
            pose_range=pose_range,
        )

        target_peg_pos_w = target_peg_pos_local + env.scene.env_origins[
            unresolved_env_ids
        ]

        target_peg_pose_w = torch.cat(
            [target_peg_pos_w, target_peg_quat_wxyz],
            dim=-1,
        )

        default_joint_pos = robot_entity.data.default_joint_pos[
            unresolved_env_ids
        ].clone()

        still_unresolved = []

        for local_index, env_id_int in enumerate(unresolved_env_ids.tolist()):

            q_joint_seed = (
                default_joint_pos[local_index]
                .detach()
                .cpu()
                .numpy()
                .copy()
            )

            if q_joint_seed.shape[0] != num_robot_joints:
                raise ValueError(
                    f"default_joint_pos has {q_joint_seed.shape[0]} joints, "
                    f"but inferred MuJoCo robot joints are {num_robot_joints}: "
                    f"{robot_joint_names}."
                )

            qpos_full = _build_full_qpos_for_mink(
                mj_model=mj_model,
                robot_joint_pos_np=q_joint_seed,
                robot_joint_qpos_indices=robot_joint_qpos_indices,
            )

            configuration = _make_mink_configuration(
                mj_model=mj_model,
                qpos_full=qpos_full,
            )

            # mink 内部也把 peg 放到安全位置。
            safe_local_np = np.asarray(
                safe_peg_pos,
                dtype=np.float64,
            )

            safe_quat_np = np.asarray(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )

            _set_entity_mocap_pose_in_data(
                mj_model=mj_model,
                mj_data=configuration.data,
                entity_name=asset_cfg.name,
                pos_local_np=safe_local_np,
                quat_wxyz_np=safe_quat_np,
            )

            mujoco.mj_forward(mj_model, configuration.data)

            # 根据本轮采样的 peg 目标位姿，计算对应 EE 目标位姿
            peg_pos_local_np = (
                target_peg_pos_local[local_index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            peg_quat_np = (
                target_peg_quat_wxyz[local_index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            target_pos_local_np, target_quat_wxyz_np = _compute_target_pose_from_peg_target(
                mj_model=mj_model,
                peg_site_name=peg_site_name,
                peg_entity_name=asset_cfg.name,
                peg_pos_local_np=peg_pos_local_np,
                peg_quat_wxyz_np=peg_quat_np,
                z_offset=float(z_offset),
            )

            frame_task = mink.FrameTask(
                frame_name=ee_site_name,
                frame_type="site",
                position_cost=float(ik_position_cost),
                orientation_cost=float(ik_orientation_cost),
                gain=1.0,
                lm_damping=float(ik_damping),
            )

            target_wxyz_xyz = np.concatenate(
                [target_quat_wxyz_np, target_pos_local_np],
                axis=0,
            )

            frame_task.set_target(
                mink.SE3(wxyz_xyz=target_wxyz_xyz)
            )

            ik_success = False
            last_pos_err = np.inf
            last_ori_err = np.inf

            for _ in range(ik_iterations):
                try:
                    vel = mink.solve_ik(
                        configuration=configuration,
                        tasks=[frame_task],
                        limits=limits,
                        dt=float(ik_dt),
                        solver=ik_solver,
                        damping=float(ik_damping),
                        safety_break=False,
                    )
                except Exception as exc:
                    print(
                        f"[IK warning] attempt={attempt + 1}, "
                        f"env_id={env_id_int}, mink.solve_ik failed: {exc}"
                    )
                    break

                configuration.integrate_inplace(vel, float(ik_dt))
                mujoco.mj_forward(mj_model, configuration.data)

                err = frame_task.compute_error(configuration)

                pos_err = np.linalg.norm(err[:3])
                ori_err = np.linalg.norm(err[3:])

                last_pos_err = pos_err
                last_ori_err = ori_err

                if pos_err < ik_pos_tol and ori_err < ik_ori_tol:
                    ik_success = True
                    break

            if not ik_success:
                print(
                    f"[IK warning] attempt={attempt + 1}, "
                    f"env_id={env_id_int}, IK not converged. "
                    f"pos_err={last_pos_err:.6f}, "
                    f"ori_err={last_ori_err:.6f}. Resampling..."
                )
                still_unresolved.append(env_id_int)
                continue

            # ------------------------------------------------
            # 检查机械臂自身是否碰撞
            # ------------------------------------------------
            mujoco.mj_forward(mj_model, configuration.data)

            has_self_collision = _has_robot_self_collision(
                mj_model=mj_model,
                mj_data=configuration.data,
                robot_geom_names=robot_self_collision_geom_names,
                min_penetration=float(self_collision_min_penetration),
            )

            if has_self_collision:
                print(
                    f"[IK warning] attempt={attempt + 1}, "
                    f"env_id={env_id_int}, IK solution has self-collision. "
                    "Resampling target pose..."
                )
                still_unresolved.append(env_id_int)
                continue

            # 当前 env 成功
            qpos_solution_full = configuration.q.copy()

            q_joint_solution = qpos_solution_full[
                robot_joint_qpos_indices
            ].astype(np.float32)

            accepted_joint_pos[env_id_int] = q_joint_solution
            accepted_peg_pose_w[env_id_int] = target_peg_pose_w[
                local_index
            ].detach().clone()

        unresolved_env_ids = torch.tensor(
            still_unresolved,
            device=env.device,
            dtype=torch.long,
        )

    # ----------------------------------------------------
    # 3. 写入所有成功 env 的机械臂关节角和 peg 目标位姿
    # ----------------------------------------------------
    success_env_ids = [
        int(eid) for eid in all_env_ids.tolist()
        if int(eid) in accepted_joint_pos
    ]

    failed_env_ids = [
        int(eid) for eid in all_env_ids.tolist()
        if int(eid) not in accepted_joint_pos
    ]

    if len(success_env_ids) > 0:
        success_env_tensor = torch.tensor(
            success_env_ids,
            device=env.device,
            dtype=torch.long,
        )

        solved_joint_pos = torch.tensor(
            np.stack(
                [accepted_joint_pos[eid] for eid in success_env_ids],
                axis=0,
            ),
            device=env.device,
            dtype=torch.float32,
        )

        solved_joint_vel = torch.zeros_like(solved_joint_pos)

        robot_entity.write_joint_state_to_sim(
            solved_joint_pos,
            solved_joint_vel,
            env_ids=success_env_tensor,
        )

        success_peg_pose_w = torch.stack(
            [accepted_peg_pose_w[eid] for eid in success_env_ids],
            dim=0,
        ).to(device=env.device, dtype=torch.float32)

        peg_entity.write_mocap_pose_to_sim(
            success_peg_pose_w,
            env_ids=success_env_tensor,
        )

    # 失败的 env 保持 peg 在安全位置，机器人保持默认姿态
    if len(failed_env_ids) > 0:
        failed_env_tensor = torch.tensor(
            failed_env_ids,
            device=env.device,
            dtype=torch.long,
        )

        failed_safe_pos_local = torch.tensor(
            safe_peg_pos,
            device=env.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(len(failed_env_ids), -1)

        failed_safe_pos_w = failed_safe_pos_local + env.scene.env_origins[
            failed_env_tensor
        ]

        failed_safe_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=env.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(len(failed_env_ids), -1)

        failed_safe_pose = torch.cat(
            [failed_safe_pos_w, failed_safe_quat],
            dim=-1,
        )

        peg_entity.write_mocap_pose_to_sim(
            failed_safe_pose,
            env_ids=failed_env_tensor,
        )

        fallback_joint_pos = robot_entity.data.default_joint_pos[
            failed_env_tensor
        ].clone()

        fallback_joint_vel = torch.zeros_like(fallback_joint_pos)

        robot_entity.write_joint_state_to_sim(
            fallback_joint_pos,
            fallback_joint_vel,
            env_ids=failed_env_tensor,
        )

    env.sim.forward()

    if len(failed_env_ids) > 0:
        print(
            f"[reset IK warning] Failed env_ids after "
            f"{max_resample_attempts} resampling attempts: {failed_env_ids}"
        )
        return 1

    return 0


# observation

from mjlab.utils.lab_api.math import quat_inv, quat_mul


def target_pose_ee(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("UR5e"),
) -> torch.Tensor:
    """
    Relative goal pose in end-effector frame.

    Returns:
        Tensor of shape (num_envs, 7):
        [target_pos_ee(3), target_quat_err(4)]
    """
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, ReachTargetCommand):
        raise TypeError(
            f"Command '{command_name}' must be a ReachTargetCommand, got {type(command)}"
        )

    robot: Entity = env.scene[asset_cfg.name]
    (site_ids, site_names) = robot.find_sites(_EE_SITE_NAME)

    if site_names[0] == _EE_SITE_NAME:
        ee_site_id = site_ids[0]
        ee_pos_w = robot.data.site_pos_w[:, ee_site_id]
        ee_quat_w = robot.data.site_quat_w[:, ee_site_id]
    else:
        ee_pos_w = robot.data.site_pos_w[:, 0]
        ee_quat_w = robot.data.site_quat_w[:, 0]

    # target pose in world frame
    target_pos_w = command.target_pos
    target_quat_w = command.target_quat

    # position error: world -> ee frame
    err_pos_w = target_pos_w - ee_pos_w
    target_pos_ee = quat_apply(quat_inv(ee_quat_w), err_pos_w)

    # orientation error quaternion: current ee -> target
    quat_err = quat_mul(target_quat_w, quat_inv(ee_quat_w))

    # normalize for numerical stability
    quat_err = quat_err / torch.norm(quat_err, dim=-1, keepdim=True).clamp_min(1e-8)

    # fix quaternion sign ambiguity: enforce w >= 0
    quat_err = torch.where(quat_err[:, :1] < 0.0, -quat_err, quat_err)

    return torch.cat([target_pos_ee, quat_err], dim=-1)


############### Rewards ###############

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

def make_peg_env_cfg() -> ManagerBasedRlEnvCfg:
    """
    创建 peg 任务配置。
    """

    # Actor 观察项
    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg("UR5e", joint_names=(".*",))
            },
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={
                "asset_cfg": SceneEntityCfg("UR5e", joint_names=(".*",))
            },
        ),
        "ee_to_goal": ObservationTermCfg(
            func=target_pose_ee,
            params={
                "command_name": "reach_target",
                "asset_cfg": SceneEntityCfg("UR5e", site_names=(_EE_SITE_NAME,)),  # Set per-robot.
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "actions": ObservationTermCfg(
            func=mdp.last_action,
        ),
    }

    # Critic 观察项
    critic_terms = {**actor_terms}

    # 观察项组
    observations: dict[str, ObservationGroupCfg] = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    # 动作项
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="UR5e",
            actuator_names=(".*",),
            scale=0.2,  # Override per-robot.
            use_default_offset=True,
        )
    }

    # 命令项
    commands: dict[str, CommandTermCfg] = {
        "reach_target": ReachTargetCommandCfg(
            resampling_time_range=(8.0, 12.0),
            debug_vis=True,
            difficulty="dynamic",
            target_position_range=ReachTargetCommandCfg.TargetPositionRangeCfg(
                x=(-0.3, 0.3),
                y=(-0.4, -0.7),
                z=(0.3, 0.7),
            ),
        )
    }

    # 事件项
    events: dict[str, EventTermCfg] = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("UR5e", joint_names=(".*",)),
            },
        ),

        "reset_peg_and_ur5e_ik": EventTermCfg(
            func=reset_peg_pose_and_ur5e_ik,
            mode="reset",
            params={
                "pose_range": {
                    "inner_radius": 0.2,
                    "outer_radius": 0.5,
                    "z_start": 0.3,
                    "z_end": 0.7,
                    "roll_range": (-0.2, 0.2),
                    "pitch_range": (-0.2, 0.2),
                    "yaw_offset_range": (-0.2, 0.2),
                },
                "z_offset": -0.1,
                "ik_iterations": 80,
                "ik_dt": 0.01,
                "ik_position_cost": 1.0,
                "ik_orientation_cost": 0.1,
                "asset_cfg": SceneEntityCfg("peg"),
                "robot_cfg": SceneEntityCfg("UR5e", joint_names=(".*",)),
            },
        ),
    }

    # 奖励项
    rewards: dict[str, RewardTermCfg] = {
        "pos_reach": RewardTermCfg(
            func=pos_reach_reward,
            weight=1.0,
            params={
                "command_name": "reach_target",
                "std": 0.1,
            },
        ),
        "quat_reach": RewardTermCfg(
            func=quat_reach_reward,
            weight=1.0,
            params={
                "command_name": "reach_target",
                "quat_std": 0.5,
                "pos_std": 0.1,
            },
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2, 
            weight=-0.01
        ),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={
                "asset_cfg": SceneEntityCfg("UR5e", joint_names=(".*",))
            },
        ),
        # "joint_vel_hinge": RewardTermCfg(
        #     func=manipulation_mdp.joint_velocity_hinge_penalty,
        #     weight=-0.01,
        #     params={
        #         "max_vel": 0.5,
        #         "asset_cfg": SceneEntityCfg("UR5e", joint_names=(".*",)),
        #     },
        # ),
    }

    # 终止条件
    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out, time_out=True
        ),
        # "ee_ground_collision": TerminationTermCfg(
        #     func=manipulation_mdp.illegal_contact,
        #     params={
        #         "sensor_name": "ee_ground_collision", 
        #         "force_threshold": 10.0
        #     },
        # ),
    }

    # 随时间变化的奖励权重
    # curriculum = {
    #     "joint_vel_hinge_weight": CurriculumTermCfg(
    #         func=manipulation_mdp.reward_curriculum,
    #         params={
    #             "reward_name": "joint_vel_hinge",
    #             "stages": [
    #             {"step": 0, "weight": -0.01},
    #             {"step": 500 * 24, "weight": -0.1},
    #             {"step": 1000 * 24, "weight": -1.0},
    #             ],
    #         },
    #     ),
    # }

############### Environment ###############

    viewer = ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="UR5e",
            body_name="",  # Set per-robot.
            distance=1.5,
            elevation=-5.0,
            azimuth=120.0,
        )
    sim = SimulationCfg(
            # nconmax=55,
            njmax=600,
            mujoco=MujocoCfg(
                timestep=0.001,
                iterations=10,
                ls_iterations=20,
                impratio=10,
            ),
        )
    

    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        # curriculum=curriculum,
        viewer=viewer,
        sim=sim,
        decimation=4,
        episode_length_s=20.0,
    )


def peg_env_0_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_peg_env_cfg()
    if play:
        cfg.episode_length_s = int(1e10)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}
    return cfg




def peg_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="peg_in_hole",
        logger='tensorboard',

        save_interval=200,
        num_steps_per_env=24,
        max_iterations=10_000,
    )