import mujoco
import numpy as np

xml_path = "UR5e.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

joint_names = [
    "ur_shoulder_pan_joint",
    "ur_shoulder_lift_joint",
    "ur_elbow_joint",
    "ur_wrist_1_joint",
    "ur_wrist_2_joint",
    "ur_wrist_3_joint",
]

joint_ids = []
dof_addrs = []
for name in joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_ids.append(jid)
    dof_addrs.append(model.jnt_dofadr[jid])

samples = []
nv = model.nv
M = np.zeros((nv, nv))

for _ in range(1000):
    # 随机采样到关节范围内
    q = np.zeros(len(joint_names))
    for i, jid in enumerate(joint_ids):
        r = model.jnt_range[jid]
        q[i] = np.random.uniform(r[0], r[1])

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    for i, jid in enumerate(joint_ids):
        qpos_adr = model.jnt_qposadr[jid]
        data.qpos[qpos_adr] = q[i]

    mujoco.mj_forward(model, data)
    mujoco.mj_fullM(model, M, data.qM)

    diag_vals = [M[d, d] for d in dof_addrs]
    samples.append(diag_vals)

samples = np.array(samples)

mean_eff = samples.mean(axis=0)
median_eff = np.median(samples, axis=0)
p75_eff = np.percentile(samples, 75, axis=0)

print("mean =", mean_eff) # mean = [1.63799205 2.82793945 0.92214505 0.0501127  0.0359087  0.00633158]
print("median =", median_eff) # median = [1.15571533 2.78973384 0.92467591 0.04980063 0.03584686 0.00633158]
print("p75 =", p75_eff) # p75 = [2.35943938 4.12080429 1.0963468  0.05972353 0.03699045 0.00633158]


NATURAL_FREQ = (2.0) * 2.0 * 3.1415926535  # 自然频率
DAMPING_RATIO = (1.2)  # 阻尼比

mean_Kp = mean_eff * NATURAL_FREQ**2
mean_Kd = 2.0 * DAMPING_RATIO * mean_eff * NATURAL_FREQ
print("mean_Kp =", mean_Kp)
print("mean_Kd =", mean_Kd)

median_Kp = median_eff * NATURAL_FREQ**2
median_Kd = 2.0 * DAMPING_RATIO * median_eff * NATURAL_FREQ
print("median_Kp =", median_Kp)
print("median_Kd =", median_Kd)

p75_Kp = p75_eff * NATURAL_FREQ**2
p75_Kd = 2.0 * DAMPING_RATIO * p75_eff * NATURAL_FREQ
print("p75_Kp =", p75_Kp)
print("p75_Kd =", p75_Kd)
