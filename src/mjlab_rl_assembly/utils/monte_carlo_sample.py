#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte Carlo sampler for UR robot + peg target pose using separate XML files.

使用方式 1：直接修改 main() 中的参数，然后运行：
    python monte_carlo_two_xml_mink_sampler.py

使用方式 2：命令行参数运行：
    python monte_carlo_two_xml_mink_sampler.py \
        --robot-xml path/to/robot.xml \
        --peg-xml path/to/peg.xml \
        --out feasible_reset_dataset.npz \
        --num-samples 500 \
        --visualize

核心逻辑：
1. robot_xml 只用于机械臂 IK 和自碰撞检测；
2. peg_xml 只用于计算 PEG_SITE 相对于 peg body 的固定变换；
3. 在机械臂基座 1m 球内随机采样 peg body 位姿；
4. 根据 peg body 位姿计算 PEG_SITE 位姿；
5. UR_EE_SITE 初始目标位姿 = PEG_SITE 沿其局部 z 轴偏移 10 cm；
6. 用 mink 解初始 IK；
7. 再验证 UR_EE_SITE 能否到达 PEG_SITE；
8. 检查初始位姿和到达位姿下机械臂是否自碰撞；
9. 保存可行样本。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mink
import numpy as np


# ============================================================
# Config
# ============================================================

@dataclass
class SamplerConfig:
    robot_xml: str
    peg_xml: str
    out_path: str = "feasible_reset_dataset.npz"

    # sites
    ee_site_name: str = "UR_EE_SITE"
    peg_site_name: str = "PEG_SITE"

    # body / prefix
    robot_prefix: str = "UR5e"
    robot_geom_prefix: str = "UR5e"
    robot_base_body: str | None = None

    # peg body
    # 如果为 None，则默认使用 PEG_SITE 所在 body 作为 peg body
    peg_body_name: str | None = None

    # sampling
    num_samples: int = 500
    max_trials: int = 200000
    workspace_radius: float = 1.0
    seed: int = 0

    # 初始末端位置相对 PEG_SITE 的偏移
    # +0.10 表示沿 PEG_SITE 局部 +z 方向 10 cm
    # -0.10 表示沿 PEG_SITE 局部 -z 方向 10 cm
    ee_offset_z: float = 0.10

    # peg 姿态采样
    # "uniform_quat"：SO(3) 随机姿态
    # "euler_range"：按 roll/pitch/yaw 范围采样
    peg_orientation_mode: str = "uniform_quat"
    roll_range: tuple[float, float] = (-math.pi, math.pi)
    pitch_range: tuple[float, float] = (-math.pi, math.pi)
    yaw_range: tuple[float, float] = (-math.pi, math.pi)

    # IK
    ik_iterations: int = 120
    ik_dt: float = 0.01
    ik_solver: str = "daqp"
    ik_damping: float = 1e-4
    ik_position_cost: float = 1.0
    ik_orientation_cost: float = 0.1
    pos_tol: float = 5e-3
    ori_tol: float = 5e-2
    num_ik_seeds: int = 8

    # self collision
    use_self_collision_limit: bool = True
    self_collision_min_distance: float = 0.02
    self_collision_detection_distance: float = 0.15
    self_collision_min_penetration: float = 1e-5

    # 如果你的模型中某些相邻连杆会被 MuJoCo 报 contact，但你认为不是非法自碰撞，可在这里忽略
    ignored_self_collision_pairs: tuple[tuple[str, str], ...] = ()

    # visualization
    visualize: bool = False
    viz_max_samples: int = 20
    viz_switch_interval: float = 2.0

    # logging
    print_every: int = 20


# ============================================================
# MuJoCo basic helpers
# ============================================================

def mj_name2id_required(model, obj_type, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Cannot find object '{name}' of type {obj_type}.")
    return obj_id


def mj_id2name_safe(model, obj_type, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, int(obj_id))
    return "" if name is None else name


def mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, mat.reshape(-1))
    return quat


def quat_wxyz_to_mat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat.astype(np.float64))
    return mat.reshape(3, 3)


def quat_angle_error(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    dot = abs(float(np.dot(q1, q2)))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * math.acos(dot)


def euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    返回 wxyz 四元数。
    """
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def uniform_quat_wxyz(rng: np.random.Generator) -> np.ndarray:
    """
    SO(3) 均匀随机四元数，wxyz 顺序。
    """
    u1, u2, u3 = rng.random(3)

    qx = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    qy = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    qz = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    qw = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def sample_position_in_sphere(
    rng: np.random.Generator,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    在球体内部均匀采样。
    """
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction) + 1e-12

    r = float(radius) * (rng.random() ** (1.0 / 3.0))
    return center + r * direction


def sample_peg_orientation(
    rng: np.random.Generator,
    cfg: SamplerConfig,
) -> np.ndarray:
    if cfg.peg_orientation_mode == "uniform_quat":
        return uniform_quat_wxyz(rng)

    if cfg.peg_orientation_mode == "euler_range":
        roll = rng.uniform(cfg.roll_range[0], cfg.roll_range[1])
        pitch = rng.uniform(cfg.pitch_range[0], cfg.pitch_range[1])
        yaw = rng.uniform(cfg.yaw_range[0], cfg.yaw_range[1])
        return euler_xyz_to_quat_wxyz(roll, pitch, yaw)

    raise ValueError(
        f"Unknown peg_orientation_mode: {cfg.peg_orientation_mode}. "
        "Use 'uniform_quat' or 'euler_range'."
    )


# ============================================================
# Transform helpers
# ============================================================

def invert_transform(pos: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    T^{-1}
    """
    rot_inv = rot.T
    pos_inv = -rot_inv @ pos
    return pos_inv, rot_inv


def compose_transform(
    pos_a: np.ndarray,
    rot_a: np.ndarray,
    pos_b: np.ndarray,
    rot_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    T_ab = T_a * T_b
    """
    pos = pos_a + rot_a @ pos_b
    rot = rot_a @ rot_b
    return pos, rot


# ============================================================
# Peg XML processing
# ============================================================

def extract_peg_site_transform_in_body(
    peg_xml: str,
    peg_site_name: str,
    peg_body_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    从 peg_xml 中提取 PEG_SITE 相对于 peg body 的固定变换。

    返回：
        site_pos_in_body: [3]
        site_rot_in_body: [3,3]
        used_peg_body_name: str

    如果 peg_body_name=None，则使用 PEG_SITE 所在 body 作为 peg body。
    """
    peg_model = mujoco.MjModel.from_xml_path(peg_xml)
    peg_data = mujoco.MjData(peg_model)
    mujoco.mj_forward(peg_model, peg_data)

    site_id = mj_name2id_required(
        peg_model,
        mujoco.mjtObj.mjOBJ_SITE,
        peg_site_name,
    )

    if peg_body_name is None:
        body_id = int(peg_model.site_bodyid[site_id])
        used_body_name = mj_id2name_safe(
            peg_model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )
    else:
        body_id = mj_name2id_required(
            peg_model,
            mujoco.mjtObj.mjOBJ_BODY,
            peg_body_name,
        )
        used_body_name = peg_body_name

    body_pos = peg_data.xpos[body_id].copy()
    body_rot = peg_data.xmat[body_id].reshape(3, 3).copy()

    site_pos = peg_data.site_xpos[site_id].copy()
    site_rot = peg_data.site_xmat[site_id].reshape(3, 3).copy()

    body_inv_pos, body_inv_rot = invert_transform(body_pos, body_rot)
    site_pos_in_body, site_rot_in_body = compose_transform(
        body_inv_pos,
        body_inv_rot,
        site_pos,
        site_rot,
    )

    return site_pos_in_body, site_rot_in_body, used_body_name


def compute_peg_site_pose_from_body_pose(
    peg_body_pos: np.ndarray,
    peg_body_quat_wxyz: np.ndarray,
    peg_site_pos_in_body: np.ndarray,
    peg_site_rot_in_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据随机采样的 peg body 位姿，计算 PEG_SITE 世界位姿。
    """
    peg_body_rot = quat_wxyz_to_mat(peg_body_quat_wxyz)

    site_pos, site_rot = compose_transform(
        peg_body_pos,
        peg_body_rot,
        peg_site_pos_in_body,
        peg_site_rot_in_body,
    )

    site_quat = mat_to_quat_wxyz(site_rot)

    return site_pos, site_quat, site_rot


# ============================================================
# Robot joint and collision helpers
# ============================================================

def infer_robot_joint_names(
    model,
    robot_prefix: str,
) -> list[str]:
    """
    根据 prefix 自动推断机器人 1-DoF 关节。
    """
    names = []

    for jid in range(model.njnt):
        jtype = int(model.jnt_type[jid])

        if jtype not in [
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ]:
            continue

        name = mj_id2name_safe(model, mujoco.mjtObj.mjOBJ_JOINT, jid)

        if robot_prefix:
            if name.startswith(robot_prefix):
                names.append(name)
        else:
            names.append(name)

    if len(names) == 0:
        raise RuntimeError(
            f"No hinge/slide joints found with robot_prefix='{robot_prefix}'."
        )

    return names


def get_joint_ids_and_qpos_indices(
    model,
    joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = []
    qpos_indices = []

    for name in joint_names:
        jid = mj_name2id_required(model, mujoco.mjtObj.mjOBJ_JOINT, name)

        jtype = int(model.jnt_type[jid])
        if jtype not in [
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ]:
            raise ValueError(f"Joint '{name}' is not hinge/slide joint.")

        joint_ids.append(jid)
        qpos_indices.append(int(model.jnt_qposadr[jid]))

    return (
        np.asarray(joint_ids, dtype=np.int64),
        np.asarray(qpos_indices, dtype=np.int64),
    )


def infer_robot_geom_names(
    model,
    robot_geom_prefix: str,
) -> list[str]:
    names = []

    for gid in range(model.ngeom):
        name = mj_id2name_safe(model, mujoco.mjtObj.mjOBJ_GEOM, gid)

        if robot_geom_prefix:
            if name.startswith(robot_geom_prefix):
                names.append(name)
        else:
            names.append(name)

    if len(names) == 0:
        raise RuntimeError(
            f"No geoms found with robot_geom_prefix='{robot_geom_prefix}'."
        )

    return names


def sample_robot_seed_qpos(
    rng: np.random.Generator,
    model,
    joint_ids: np.ndarray,
    joint_qpos_indices: np.ndarray,
    seed_index: int,
) -> np.ndarray:
    """
    seed_index=0 使用 qpos0；
    其他 seed 在关节范围内随机采样。
    """
    qpos = model.qpos0.copy()

    if seed_index == 0:
        return qpos

    for jid, qadr in zip(joint_ids, joint_qpos_indices):
        if bool(model.jnt_limited[jid]):
            lo, hi = model.jnt_range[jid]
        else:
            lo, hi = -math.pi, math.pi

        qpos[qadr] = rng.uniform(float(lo), float(hi))

    return qpos


def site_quat_from_data(data, site_id: int) -> np.ndarray:
    site_rot = data.site_xmat[site_id].reshape(3, 3).copy()
    return mat_to_quat_wxyz(site_rot)


def compute_site_pose_error(
    data,
    site_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
) -> tuple[float, float]:
    current_pos = data.site_xpos[site_id].copy()
    current_quat = site_quat_from_data(data, site_id)

    pos_err = float(np.linalg.norm(current_pos - target_pos))
    ori_err = float(quat_angle_error(current_quat, target_quat))

    return pos_err, ori_err


def normalize_pair_name(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def has_robot_self_collision(
    model,
    data,
    robot_geom_names: list[str],
    ignored_pairs: tuple[tuple[str, str], ...] = (),
    min_penetration: float = 1e-5,
) -> bool:
    robot_set = set(robot_geom_names)
    ignored = {normalize_pair_name(a, b) for a, b in ignored_pairs}

    for i in range(data.ncon):
        con = data.contact[i]

        g1 = mj_id2name_safe(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
        g2 = mj_id2name_safe(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)

        if g1 not in robot_set or g2 not in robot_set:
            continue

        if normalize_pair_name(g1, g2) in ignored:
            continue

        if con.dist < -float(min_penetration):
            return True

    return False


# ============================================================
# Mink IK
# ============================================================

def make_mink_configuration(model, qpos: np.ndarray):
    try:
        configuration = mink.Configuration(model)
        configuration.update(qpos)
    except TypeError:
        configuration = mink.Configuration(model, qpos)
    return configuration


def solve_ik_to_site_pose(
    model,
    ee_site_name: str,
    ee_site_id: int,
    q_seed: np.ndarray,
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    robot_geom_names: list[str],
    cfg: SamplerConfig,
) -> tuple[bool, np.ndarray, float, float, bool]:
    """
    使用 mink 求解 UR_EE_SITE 到指定目标位姿的 IK。
    返回：
        success
        q_solution_full
        pos_err
        ori_err
        has_self_collision
    """
    configuration = make_mink_configuration(model, q_seed)

    frame_task = mink.FrameTask(
        frame_name=ee_site_name,
        frame_type="site",
        position_cost=float(cfg.ik_position_cost),
        orientation_cost=float(cfg.ik_orientation_cost),
        gain=1.0,
        lm_damping=float(cfg.ik_damping),
    )

    # mink.SE3 顺序为 [qw, qx, qy, qz, x, y, z]
    target_wxyz_xyz = np.concatenate(
        [
            np.asarray(target_quat_wxyz, dtype=np.float64),
            np.asarray(target_pos, dtype=np.float64),
        ],
        axis=0,
    )

    frame_task.set_target(mink.SE3(wxyz_xyz=target_wxyz_xyz))

    limits = [mink.ConfigurationLimit(model)]

    if cfg.use_self_collision_limit and len(robot_geom_names) > 0:
        # 自碰撞：同一个 geom group 放在 pair 两侧
        limits.append(
            mink.CollisionAvoidanceLimit(
                model=model,
                geom_pairs=[(robot_geom_names, robot_geom_names)],
                gain=0.95,
                minimum_distance_from_collisions=float(cfg.self_collision_min_distance),
                collision_detection_distance=float(cfg.self_collision_detection_distance),
                bound_relaxation=0.0,
            )
        )

    pos_err = np.inf
    ori_err = np.inf

    for _ in range(int(cfg.ik_iterations)):
        try:
            vel = mink.solve_ik(
                configuration=configuration,
                tasks=[frame_task],
                limits=limits,
                dt=float(cfg.ik_dt),
                solver=cfg.ik_solver,
                damping=float(cfg.ik_damping),
                safety_break=False,
            )
        except Exception:
            break

        configuration.integrate_inplace(vel, float(cfg.ik_dt))
        mujoco.mj_forward(model, configuration.data)

        pos_err, ori_err = compute_site_pose_error(
            configuration.data,
            ee_site_id,
            target_pos,
            target_quat_wxyz,
        )

        if pos_err < cfg.pos_tol and ori_err < cfg.ori_tol:
            break

    mujoco.mj_forward(model, configuration.data)

    pos_err, ori_err = compute_site_pose_error(
        configuration.data,
        ee_site_id,
        target_pos,
        target_quat_wxyz,
    )

    has_self_col = has_robot_self_collision(
        model=model,
        data=configuration.data,
        robot_geom_names=robot_geom_names,
        ignored_pairs=cfg.ignored_self_collision_pairs,
        min_penetration=cfg.self_collision_min_penetration,
    )

    success = (
        pos_err < cfg.pos_tol
        and ori_err < cfg.ori_tol
        and not has_self_col
    )

    return (
        bool(success),
        configuration.q.copy(),
        float(pos_err),
        float(ori_err),
        bool(has_self_col),
    )


# ============================================================
# Visualization
# ============================================================

def add_sphere_marker(
    viewer,
    pos: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return

    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


def visualize_samples(
    model,
    samples: list[dict],
    joint_qpos_indices: np.ndarray,
    cfg: SamplerConfig,
) -> None:
    """
    可视化调试：
    - 显示机器人初始 IK 构型；
    - 绿色球：PEG_SITE 位置；
    - 蓝色球：UR_EE_SITE 初始目标位置，即 PEG_SITE z 方向 10 cm 偏移点。
    """
    try:
        import mujoco.viewer
    except Exception as exc:
        print(f"[visualize] mujoco.viewer import failed: {exc}")
        return

    if len(samples) == 0:
        print("[visualize] no samples to visualize.")
        return

    data = mujoco.MjData(model)

    viz_samples = samples[: min(len(samples), cfg.viz_max_samples)]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        idx = 0
        last_switch_time = time.time()

        while viewer.is_running():
            now = time.time()

            if now - last_switch_time > cfg.viz_switch_interval:
                idx = (idx + 1) % len(viz_samples)
                last_switch_time = now

            s = viz_samples[idx]

            data.qpos[:] = model.qpos0
            data.qpos[joint_qpos_indices] = s["ur_joint_pos_initial"]
            mujoco.mj_forward(model, data)

            viewer.user_scn.ngeom = 0

            add_sphere_marker(
                viewer,
                s["peg_site_pos"],
                radius=0.025,
                rgba=np.array([0.0, 1.0, 0.0, 0.85]),
            )

            add_sphere_marker(
                viewer,
                s["ee_initial_target_pos"],
                radius=0.02,
                rgba=np.array([0.0, 0.25, 1.0, 0.85]),
            )

            add_sphere_marker(
                viewer,
                s["peg_body_pos"],
                radius=0.018,
                rgba=np.array([1.0, 0.5, 0.0, 0.85]),
            )

            viewer.sync()
            time.sleep(0.01)


# ============================================================
# Save
# ============================================================

def save_dataset(
    out_path: str,
    samples: list[dict],
    joint_names: list[str],
    metadata: dict,
) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    peg_body_pos = np.stack([s["peg_body_pos"] for s in samples], axis=0)
    peg_body_quat = np.stack([s["peg_body_quat_wxyz"] for s in samples], axis=0)
    peg_site_pos = np.stack([s["peg_site_pos"] for s in samples], axis=0)
    peg_site_quat = np.stack([s["peg_site_quat_wxyz"] for s in samples], axis=0)

    ee_init_pos = np.stack([s["ee_initial_target_pos"] for s in samples], axis=0)
    ee_init_quat = np.stack([s["ee_initial_target_quat_wxyz"] for s in samples], axis=0)

    ee_reach_pos = np.stack([s["ee_reach_target_pos"] for s in samples], axis=0)
    ee_reach_quat = np.stack([s["ee_reach_target_quat_wxyz"] for s in samples], axis=0)

    q_init = np.stack([s["ur_joint_pos_initial"] for s in samples], axis=0)
    q_reach = np.stack([s["ur_joint_pos_reach"] for s in samples], axis=0)

    np.savez_compressed(
        path,
        peg_body_pos=peg_body_pos,
        peg_body_quat_wxyz=peg_body_quat,
        peg_body_pose_wxyz_xyz=np.concatenate([peg_body_quat, peg_body_pos], axis=1),
        peg_site_pos=peg_site_pos,
        peg_site_quat_wxyz=peg_site_quat,
        ee_initial_target_pos=ee_init_pos,
        ee_initial_target_quat_wxyz=ee_init_quat,
        ee_reach_target_pos=ee_reach_pos,
        ee_reach_target_quat_wxyz=ee_reach_quat,
        ur_joint_pos_initial=q_init,
        ur_joint_pos_reach=q_reach,
        joint_names=np.asarray(joint_names, dtype=object),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=object),
    )

    csv_path = path.with_suffix(".csv")

    header = (
        ["peg_x", "peg_y", "peg_z", "peg_qw", "peg_qx", "peg_qy", "peg_qz"]
        + [f"q_init_{name}" for name in joint_names]
        + [f"q_reach_{name}" for name in joint_names]
        + [
            "peg_site_x", "peg_site_y", "peg_site_z",
            "ee_init_x", "ee_init_y", "ee_init_z",
        ]
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for s in samples:
            row = (
                list(s["peg_body_pos"])
                + list(s["peg_body_quat_wxyz"])
                + list(s["ur_joint_pos_initial"])
                + list(s["ur_joint_pos_reach"])
                + list(s["peg_site_pos"])
                + list(s["ee_initial_target_pos"])
            )
            writer.writerow(row)

    print(f"[save] npz: {path}")
    print(f"[save] csv: {csv_path}")


# ============================================================
# Main sampling logic
# ============================================================

def run_sampler(cfg: SamplerConfig) -> list[dict]:
    rng = np.random.default_rng(cfg.seed)

    robot_model = mujoco.MjModel.from_xml_path(cfg.robot_xml)
    robot_data = mujoco.MjData(robot_model)
    mujoco.mj_forward(robot_model, robot_data)

    ee_site_id = mj_name2id_required(
        robot_model,
        mujoco.mjtObj.mjOBJ_SITE,
        cfg.ee_site_name,
    )

    if cfg.robot_base_body is not None:
        base_body_id = mj_name2id_required(
            robot_model,
            mujoco.mjtObj.mjOBJ_BODY,
            cfg.robot_base_body,
        )
        base_center = robot_data.xpos[base_body_id].copy()
    else:
        base_center = np.zeros(3, dtype=np.float64)

    joint_names = infer_robot_joint_names(robot_model, cfg.robot_prefix)
    joint_ids, joint_qpos_indices = get_joint_ids_and_qpos_indices(
        robot_model,
        joint_names,
    )

    robot_geom_names = infer_robot_geom_names(
        robot_model,
        cfg.robot_geom_prefix,
    )

    (
        peg_site_pos_in_body,
        peg_site_rot_in_body,
        used_peg_body_name,
    ) = extract_peg_site_transform_in_body(
        peg_xml=cfg.peg_xml,
        peg_site_name=cfg.peg_site_name,
        peg_body_name=cfg.peg_body_name,
    )

    print("========== Monte Carlo Sampler ==========")
    print(f"robot_xml: {cfg.robot_xml}")
    print(f"peg_xml:   {cfg.peg_xml}")
    print(f"ee_site:   {cfg.ee_site_name}")
    print(f"peg_site:  {cfg.peg_site_name}")
    print(f"peg_body:  {used_peg_body_name}")
    print(f"base_center: {base_center}")
    print(f"joint_names: {joint_names}")
    print(f"num_robot_geoms: {len(robot_geom_names)}")
    print(f"workspace_radius: {cfg.workspace_radius}")
    print(f"ee_offset_z: {cfg.ee_offset_z}")
    print("=========================================")

    samples: list[dict] = []

    total_trials = 0
    reject_init_ik = 0
    reject_reach_ik = 0
    reject_self_collision = 0

    while len(samples) < cfg.num_samples and total_trials < cfg.max_trials:
        total_trials += 1

        # ------------------------------------------------
        # 1. 采样 peg body 位姿
        # ------------------------------------------------
        peg_body_pos = sample_position_in_sphere(
            rng=rng,
            center=base_center,
            radius=cfg.workspace_radius,
        )

        peg_body_quat = sample_peg_orientation(rng, cfg)

        # ------------------------------------------------
        # 2. 根据 peg body 位姿计算 PEG_SITE 位姿
        # ------------------------------------------------
        peg_site_pos, peg_site_quat, peg_site_rot = compute_peg_site_pose_from_body_pose(
            peg_body_pos=peg_body_pos,
            peg_body_quat_wxyz=peg_body_quat,
            peg_site_pos_in_body=peg_site_pos_in_body,
            peg_site_rot_in_body=peg_site_rot_in_body,
        )

        # 初始位姿：UR_EE_SITE 与 PEG_SITE 沿 PEG_SITE 局部 z 轴相距 10 cm
        ee_initial_target_pos = peg_site_pos + peg_site_rot @ np.array(
            [0.0, 0.0, cfg.ee_offset_z],
            dtype=np.float64,
        )
        ee_initial_target_quat = peg_site_quat.copy()

        # 到达位姿：UR_EE_SITE 到达 PEG_SITE
        ee_reach_target_pos = peg_site_pos.copy()
        ee_reach_target_quat = peg_site_quat.copy()

        # ------------------------------------------------
        # 3. 初始 IK：UR_EE_SITE 到偏移 10cm 的初始目标点
        # ------------------------------------------------
        init_success = False
        q_initial_full = None
        init_self_col = False

        for seed_index in range(cfg.num_ik_seeds):
            q_seed = sample_robot_seed_qpos(
                rng=rng,
                model=robot_model,
                joint_ids=joint_ids,
                joint_qpos_indices=joint_qpos_indices,
                seed_index=seed_index,
            )

            (
                init_success,
                q_candidate_full,
                init_pos_err,
                init_ori_err,
                init_self_col,
            ) = solve_ik_to_site_pose(
                model=robot_model,
                ee_site_name=cfg.ee_site_name,
                ee_site_id=ee_site_id,
                q_seed=q_seed,
                target_pos=ee_initial_target_pos,
                target_quat_wxyz=ee_initial_target_quat,
                robot_geom_names=robot_geom_names,
                cfg=cfg,
            )

            if init_success:
                q_initial_full = q_candidate_full
                break

        if not init_success or q_initial_full is None:
            reject_init_ik += 1
            if init_self_col:
                reject_self_collision += 1
            continue

        # ------------------------------------------------
        # 4. 验证能否到达 PEG_SITE
        # ------------------------------------------------
        (
            reach_success,
            q_reach_full,
            reach_pos_err,
            reach_ori_err,
            reach_self_col,
        ) = solve_ik_to_site_pose(
            model=robot_model,
            ee_site_name=cfg.ee_site_name,
            ee_site_id=ee_site_id,
            q_seed=q_initial_full,
            target_pos=ee_reach_target_pos,
            target_quat_wxyz=ee_reach_target_quat,
            robot_geom_names=robot_geom_names,
            cfg=cfg,
        )

        if not reach_success:
            reject_reach_ik += 1
            if reach_self_col:
                reject_self_collision += 1
            continue

        # ------------------------------------------------
        # 5. 保存可行样本
        # ------------------------------------------------
        q_initial = q_initial_full[joint_qpos_indices].astype(np.float32)
        q_reach = q_reach_full[joint_qpos_indices].astype(np.float32)

        samples.append(
            {
                "peg_body_pos": peg_body_pos.astype(np.float32),
                "peg_body_quat_wxyz": peg_body_quat.astype(np.float32),
                "peg_site_pos": peg_site_pos.astype(np.float32),
                "peg_site_quat_wxyz": peg_site_quat.astype(np.float32),
                "ee_initial_target_pos": ee_initial_target_pos.astype(np.float32),
                "ee_initial_target_quat_wxyz": ee_initial_target_quat.astype(np.float32),
                "ee_reach_target_pos": ee_reach_target_pos.astype(np.float32),
                "ee_reach_target_quat_wxyz": ee_reach_target_quat.astype(np.float32),
                "ur_joint_pos_initial": q_initial,
                "ur_joint_pos_reach": q_reach,
            }
        )

        if len(samples) % cfg.print_every == 0:
            accept_rate = len(samples) / max(total_trials, 1)
            print(
                f"[progress] accepted={len(samples)}/{cfg.num_samples}, "
                f"trials={total_trials}, "
                f"accept_rate={accept_rate:.4f}, "
                f"reject_init_ik={reject_init_ik}, "
                f"reject_reach_ik={reject_reach_ik}, "
                f"reject_self_collision={reject_self_collision}"
            )

    if len(samples) == 0:
        raise RuntimeError(
            "No feasible samples found. "
            "请检查 site 名称、joint prefix、geom prefix，或放宽姿态范围/误差阈值。"
        )

    metadata = {
        "robot_xml": cfg.robot_xml,
        "peg_xml": cfg.peg_xml,
        "ee_site_name": cfg.ee_site_name,
        "peg_site_name": cfg.peg_site_name,
        "used_peg_body_name": used_peg_body_name,
        "robot_prefix": cfg.robot_prefix,
        "robot_geom_prefix": cfg.robot_geom_prefix,
        "robot_base_body": cfg.robot_base_body,
        "base_center": base_center.tolist(),
        "workspace_radius": cfg.workspace_radius,
        "ee_offset_z": cfg.ee_offset_z,
        "num_requested": cfg.num_samples,
        "num_accepted": len(samples),
        "total_trials": total_trials,
        "accept_rate": len(samples) / max(total_trials, 1),
        "reject_init_ik": reject_init_ik,
        "reject_reach_ik": reject_reach_ik,
        "reject_self_collision": reject_self_collision,
        "joint_names": joint_names,
        "pos_tol": cfg.pos_tol,
        "ori_tol": cfg.ori_tol,
        "ik_iterations": cfg.ik_iterations,
        "ik_dt": cfg.ik_dt,
        "ik_solver": cfg.ik_solver,
    }

    save_dataset(
        out_path=cfg.out_path,
        samples=samples,
        joint_names=joint_names,
        metadata=metadata,
    )

    print("========== Final Statistics ==========")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("======================================")

    if cfg.visualize:
        visualize_samples(
            model=robot_model,
            samples=samples,
            joint_qpos_indices=joint_qpos_indices,
            cfg=cfg,
        )

    return samples


# ============================================================
# Argparse and manual main
# ============================================================

def config_from_args() -> SamplerConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--robot-xml", type=str, required=True)
    parser.add_argument("--peg-xml", type=str, required=True)
    parser.add_argument("--out", type=str, default="feasible_reset_dataset.npz")

    parser.add_argument("--ee-site", type=str, default="UR_EE_SITE")
    parser.add_argument("--peg-site", type=str, default="PEG_SITE")

    parser.add_argument("--robot-prefix", type=str, default="UR5e")
    parser.add_argument("--robot-geom-prefix", type=str, default="UR5e")
    parser.add_argument("--robot-base-body", type=str, default=None)
    parser.add_argument("--peg-body", type=str, default=None)

    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--max-trials", type=int, default=200000)
    parser.add_argument("--workspace-radius", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--ee-offset-z", type=float, default=0.10)

    parser.add_argument(
        "--peg-orientation-mode",
        type=str,
        default="uniform_quat",
        choices=["uniform_quat", "euler_range"],
    )

    parser.add_argument("--ik-iterations", type=int, default=120)
    parser.add_argument("--ik-dt", type=float, default=0.01)
    parser.add_argument("--ik-solver", type=str, default="daqp")
    parser.add_argument("--ik-damping", type=float, default=1e-4)
    parser.add_argument("--ik-position-cost", type=float, default=1.0)
    parser.add_argument("--ik-orientation-cost", type=float, default=0.1)
    parser.add_argument("--pos-tol", type=float, default=5e-3)
    parser.add_argument("--ori-tol", type=float, default=5e-2)
    parser.add_argument("--num-ik-seeds", type=int, default=8)

    parser.add_argument("--no-self-collision-limit", action="store_true")
    parser.add_argument("--self-collision-min-distance", type=float, default=0.02)
    parser.add_argument("--self-collision-detection-distance", type=float, default=0.15)
    parser.add_argument("--self-collision-min-penetration", type=float, default=1e-5)

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--viz-max-samples", type=int, default=20)
    parser.add_argument("--viz-switch-interval", type=float, default=2.0)
    parser.add_argument("--print-every", type=int, default=20)

    args = parser.parse_args()

    return SamplerConfig(
        robot_xml=args.robot_xml,
        peg_xml=args.peg_xml,
        out_path=args.out,
        ee_site_name=args.ee_site,
        peg_site_name=args.peg_site,
        robot_prefix=args.robot_prefix,
        robot_geom_prefix=args.robot_geom_prefix,
        robot_base_body=args.robot_base_body,
        peg_body_name=args.peg_body,
        num_samples=args.num_samples,
        max_trials=args.max_trials,
        workspace_radius=args.workspace_radius,
        seed=args.seed,
        ee_offset_z=args.ee_offset_z,
        peg_orientation_mode=args.peg_orientation_mode,
        ik_iterations=args.ik_iterations,
        ik_dt=args.ik_dt,
        ik_solver=args.ik_solver,
        ik_damping=args.ik_damping,
        ik_position_cost=args.ik_position_cost,
        ik_orientation_cost=args.ik_orientation_cost,
        pos_tol=args.pos_tol,
        ori_tol=args.ori_tol,
        num_ik_seeds=args.num_ik_seeds,
        use_self_collision_limit=not args.no_self_collision_limit,
        self_collision_min_distance=args.self_collision_min_distance,
        self_collision_detection_distance=args.self_collision_detection_distance,
        self_collision_min_penetration=args.self_collision_min_penetration,
        visualize=args.visualize,
        viz_max_samples=args.viz_max_samples,
        viz_switch_interval=args.viz_switch_interval,
        print_every=args.print_every,
    )


def main() -> None:
    """
    手动参数启动入口。

    你可以直接在这里修改路径和参数，然后运行：
        python monte_carlo_two_xml_mink_sampler.py
    """

    cfg = SamplerConfig(
        # ===============================
        # 修改成你的两个 XML 路径
        # ===============================
        robot_xml="./src/mjlab_rl_assembly/mjcf/UR5e.xml",
        peg_xml="./src/mjlab_rl_assembly/mjcf/peg.xml",

        out_path="./src/mjlab_rl_assembly/utils/reset_dataset.npz",

        # ===============================
        # site 名称
        # ===============================
        ee_site_name="_hole",
        peg_site_name="_peg",

        # ===============================
        # 机械臂命名前缀
        # 如果你的 joint/geom 名称不是 UR5e/xxx，
        # 这里要改成实际前缀。
        # ===============================
        robot_prefix="ur_",
        robot_geom_prefix="COLLISION_",

        # 如果机械臂基座 body 不是世界原点，可以填实际 body 名称
        robot_base_body="ur_base_link",

        # 如果 peg XML 中 PEG_SITE 不直接挂在 peg 本体上，
        # 建议显式填写 peg body 名称。
        peg_body_name="peg",

        # ===============================
        # 采样数量
        # ===============================
        num_samples=10_000,
        max_trials=1_000_000,
        workspace_radius=1.0,
        seed=0,

        # 初始 UR_EE_SITE 与 PEG_SITE 沿 PEG_SITE 局部 z 轴相距 10 cm
        ee_offset_z=-0.10,

        # peg 姿态采样
        # "uniform_quat" 或 "euler_range"
        peg_orientation_mode="uniform_quat",

        # 如果使用 euler_range，则这些范围生效
        roll_range=(-0.2, 0.2),
        pitch_range=(-0.2, 0.2),
        yaw_range=(-math.pi, math.pi),

        # ===============================
        # IK 参数
        # ===============================
        ik_iterations=120,
        ik_dt=0.01,
        ik_solver="daqp",
        ik_damping=1e-4,
        ik_position_cost=1.0,
        ik_orientation_cost=0.1,
        pos_tol=5e-3,
        ori_tol=5e-2,
        num_ik_seeds=8,

        # ===============================
        # 自碰撞检测
        # ===============================
        use_self_collision_limit=True,
        self_collision_min_distance=0.02,
        self_collision_detection_distance=0.15,
        self_collision_min_penetration=1e-5,

        # 如果有需要忽略的相邻连杆接触对，在这里填：
        # ignored_self_collision_pairs=(
        #     ("UR5e/COLLISION_xxx", "UR5e/COLLISION_yyy"),
        # ),
        ignored_self_collision_pairs=(),

        # ===============================
        # 可视化调试
        # ===============================
        visualize=False,
        viz_max_samples=20,
        viz_switch_interval=2.0,

        print_every=20,
    )

    run_sampler(cfg)


if __name__ == "__main__":
    # 有命令行参数时走 argparse；
    # 没有命令行参数时走 main() 手动配置。
    if len(sys.argv) > 1:
        run_sampler(config_from_args())
    else:
        main()