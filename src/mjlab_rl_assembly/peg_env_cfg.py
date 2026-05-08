from __future__ import annotations


import torch


# mjlab core
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.entity import Entity
from mjlab.sensor import BuiltinSensorCfg, ObjRef # 内置传感器
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.envs.mdp.actions import DifferentialIKActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg # 奖励函数
from mjlab.managers.termination_manager import TerminationTermCfg # 终止条件
from mjlab.managers.curriculum_manager import CurriculumTermCfg # 课程管理


from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab.sim import MujocoCfg, SimulationCfg

from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul


# rl
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


# mdp
from mjlab.envs.mdp.observations import joint_pos_rel, joint_vel_rel, last_action
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.envs.mdp.rewards import action_rate_l2, joint_pos_limits, is_alive
from mjlab.envs.mdp.terminations import time_out

from mjlab_rl_assembly.cfg.scence import get_peg_entity_cfg, get_ur5e_entity_cfg
from mjlab_rl_assembly.cfg.commands import ReachTargetCommandCfg
from mjlab_rl_assembly.cfg.events import (
    reset_peg_pose_and_ur5e_ik,
    reset_peg_pose_and_ur5e_ik_from_dataset,
)
from mjlab_rl_assembly.cfg.rewards import (
    pos_reach_reward,
    quat_reach_reward,
    align_stage_reward,
    insert_stage_reward,
    ft_penalty,
)
from mjlab_rl_assembly.cfg.observations import target_pose_ee, filtered_force_torque, get_stage
from mjlab_rl_assembly.cfg.terminations import success_peg_in_hole, failure_peg_in_hole, ft_exceed_limit
from mjlab_rl_assembly.cfg.utils import check_finite_tensor
from mjlab_rl_assembly.cfg.constants import (
    EE_SITE_NAME,
    PEG_SITE_NAME,
    UR5E_ENTITY_NAME,
    PEG_ENTITY_NAME,
    FORCE_SENSOR_NAME,
    TORQUE_SENSOR_NAME,
)



############### Rewards ###############

def make_peg_env_cfg() -> ManagerBasedRlEnvCfg:
    """
    创建 peg 任务配置。
    """
    # Scene 配置
    scene = SceneCfg(
            num_envs=1,
            env_spacing=1.0,
            terrain=TerrainEntityCfg(terrain_type="plane"), # 地形
            entities={
                UR5E_ENTITY_NAME: get_ur5e_entity_cfg(), # 机械臂
                PEG_ENTITY_NAME: get_peg_entity_cfg()  # 机械臂端
            }, 
            # sensors=(
            #     BuiltinSensorCfg(
            #         name="ee_force_sensor",
            #         sensor_type="force",
            #         obj=ObjRef(type="site", name=FORCE_SENSOR_NAME,entity=UR5E_ENTITY_NAME),
            #     ),
            #     BuiltinSensorCfg(
            #         name="ee_torque_sensor",
            #         sensor_type="torque",
            #         obj=ObjRef(type="site", name=TORQUE_SENSOR_NAME,entity=UR5E_ENTITY_NAME),
            #     ),
            # ),
        )
    
    def joint_pos_rel_debug(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg):
        obs = joint_pos_rel(env, asset_cfg=asset_cfg)
        return check_finite_tensor("joint_pos_rel", obs)
    def target_pose_ee_debug(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg):
        obs = target_pose_ee(env, command_name, asset_cfg)
        return check_finite_tensor("target_pose_ee", obs)
    def last_action_debug(env: ManagerBasedRlEnv):
        obs = last_action(env)
        return check_finite_tensor("last_action", obs)
    # Actor 观察项
    actor_terms = {
        # "joint_pos": ObservationTermCfg(
        #     func=joint_pos_rel_debug,
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        #     params={
        #         "asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, joint_names=(".*",))
        #     },
        # ),
        # "joint_vel": ObservationTermCfg(
        #     func=joint_vel_rel,
        #     noise=Unoise(n_min=-1.5, n_max=1.5),
        #     params={
        #         "asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, joint_names=(".*",))
        #     },
        # ),
        "ee_to_goal": ObservationTermCfg(
            func=target_pose_ee,
            params={
                "command_name": "reach_target",
                "asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, site_names=(EE_SITE_NAME,)),  # Set per-robot.
            },
            # noise=Unoise(n_min=-0.002, n_max=0.002),
        ),
        "ft_sensor": ObservationTermCfg(
            func=lambda env: filtered_force_torque(env ,alpha=0.2),
        ),
        "actions": ObservationTermCfg(
            func=last_action,
        ),
        "stage": ObservationTermCfg(
            func=get_stage,
            params={
                "command_name": "reach_target",
            },
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
        "ee_delta_pose": DifferentialIKActionCfg(
            entity_name=UR5E_ENTITY_NAME,

            # 控制 UR5e 的全部关节位置执行器
            actuator_names=(".*",),

            # 使用末端 site 作为 IK 控制对象
            frame_type="site",
            frame_name=EE_SITE_NAME,

            # True 表示 action 是相对当前末端位姿的增量
            # action = [dx, dy, dz, dRx, dRy, dRz]
            use_relative_mode=True,

            # 策略输出动作缩放
            # 例如 action 前三维为 [-1,1] 时，对应末端位置增量约 ±1 cm
            delta_pos_scale=0.002,

            # 后三维为姿态增量，单位 rad
            # 例如 ±0.05 rad，约 ±2.9 deg
            delta_ori_scale=0.0005,

            # DLS IK 阻尼
            damping=0.05,

            # 每次 IK 求解允许的最大关节增量，单位 rad/step
            max_dq=0.2,

            # 位姿误差权重
            position_weight=1.0,
            orientation_weight=1.0,

            # 软关节限位回避
            joint_limit_weight=0.05,

            # 姿态正则项，避免冗余自由度乱动
            posture_weight=0.005,
            posture_target={
                ".*": 0.0,
            },
        )
    }

    # 命令项
    commands: dict[str, CommandTermCfg] = {
        "reach_target": ReachTargetCommandCfg(
            resampling_time_range=(8.0, 12.0),
            debug_vis=True,
            align_pos_tolerance = 0.01,
            align_quat_tolerance = 0.05,
            insert_pos_tolerance = 0.005,
            insert_quat_tolerance = 0.02,
            failure_pos_tolerance = 0.2,
        )
    }

    # 事件项
    events: dict[str, EventTermCfg] = {
        "reset_base": EventTermCfg(
            func=reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, joint_names=(".*",)),
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
                    "pitch_range": (-1.0, 1.0),
                    "yaw_offset_range": (-0.2, 0.2),
                },
                "z_offset": -0.1,
                "ik_iterations": 80,
                "ik_dt": 0.01,
                "ik_position_cost": 1.0,
                "ik_orientation_cost": 0.1,
                "asset_cfg": SceneEntityCfg(PEG_ENTITY_NAME),
                "robot_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, joint_names=(".*",)),
            },
        ),
        # "reset_peg_and_ur5e_ik_from_dataset": EventTermCfg(
        #     func=reset_peg_pose_and_ur5e_ik_from_dataset,
        #     mode="reset",
        #     params={
        #         "dataset_path": "src/mjlab_rl_assembly/utils/reset_dataset.npz",
        #         "qpos_noise_std": 0.01,
        #     },
        # ),
    }

    # 奖励项
    rewards: dict[str, RewardTermCfg] = {
        "align_reward": RewardTermCfg( # 对齐奖励
            func=align_stage_reward,
            weight=1.0,
            params={
                "command_name": "reach_target",
                "quat_std": 0.1,
                "pos_std": 0.01,
            },
        ),
        "insert_reward": RewardTermCfg( # 插入奖励
            func=insert_stage_reward,
            weight=1.0,
            params={
                "command_name": "reach_target",
                "quat_std": 0.05,
                "pos_std": 0.005,
            },
        ),
        "pos_reach_macro": RewardTermCfg( # 位置奖励
            func=pos_reach_reward,
            weight=1.0,
            params={
                "command_name": "reach_target",
                "std": 0.1,
            },
        ),
        "quat_reach_macro": RewardTermCfg( # 姿态奖励
            func=quat_reach_reward,
            weight=1.0,
            params={
                "command_name": "reach_target",
                "quat_std": 0.5,
                "pos_std": 0.1,
            },
        ),
        "action_rate_l2": RewardTermCfg( # 动作率惩罚
            func=action_rate_l2, 
            weight=-0.01
        ),
        "joint_pos_limits": RewardTermCfg( # 关节位置惩罚
            func=joint_pos_limits,
            weight=0,#-10.0,
            params={
                "asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, joint_names=(".*",))
            },
        ),
        "run_time": RewardTermCfg( # 运行时间惩罚
            func=is_alive,
            weight=-0.1,
        ),
        "ft_penalty": RewardTermCfg( # 力矩传感器惩罚
            func=ft_penalty,
            weight=-0.5,
            params={
                "alpha": 0.2,
                "force_std": 30.0,
                "torque_std": 15.0,
            },
        ),
    }

    # 终止条件
    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(
            func=time_out, time_out=True
        ),
        "success_peg_in_hole": TerminationTermCfg(
            func=success_peg_in_hole,
            params={
                "command_name": "reach_target",
            },
        ),
        "failure_peg_in_hole": TerminationTermCfg(
            func=failure_peg_in_hole,
            params={
                "command_name": "reach_target",
            },
        ),
        "ft_exceed_limit": TerminationTermCfg(
            func=ft_exceed_limit,
            params={
                "force_limit": 60.0,
                "torque_limit": 30.0,
            },
        ),
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
            entity_name=UR5E_ENTITY_NAME,
            body_name="",  # Set per-robot.
            distance=1.5,
            elevation=-5.0,
            azimuth=120.0,
        )
    sim = SimulationCfg(
            # nconmax=55,
            njmax=1000,
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
        episode_length_s=10.0,
    )


def peg_env_0_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_peg_env_cfg()
    if play:
        # cfg.episode_length_s = int(1e10)
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
        max_iterations=5_000,
    )