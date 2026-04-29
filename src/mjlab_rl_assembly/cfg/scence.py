from pathlib import Path


import mujoco


from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator.xml_actuator import XmlActuatorCfg




# mjcf 文件路径
_UE5E_MJCF: Path = "./src/mjlab_rl_assembly/mjcf/UR5e.xml"
_PEG_MJCF: Path = "./src/mjlab_rl_assembly/mjcf/peg.xml"



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

def get_ur5e_entity_cfg() -> EntityCfg:
    """
    创建机械臂实体配置。
    """
    return EntityCfg(
        spec_fn=_get_ur5e_spec,
        articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=_UE5E_ACTUATOR_JOINTS),)),
        init_state=_UE5E_INIT_STATE,
    )



# peg 实体配置

def _get_peg_spec() -> mujoco.MjSpec:
    """
    从 mjcf 文件路径加载 mujoco 模型。
    """
    spec = mujoco.MjSpec.from_file(str(_PEG_MJCF))
    # Set the peg body as mocap to allow pose control
    spec.worldbody.first_body().mocap = True
    return spec

def get_peg_entity_cfg() -> EntityCfg:
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


