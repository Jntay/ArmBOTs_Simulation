import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

from ARM_IK_Solver import solve_ik, forward_kinematics, plot_arm_3d

# ─────────────────────────────────────────────
#  Target Position (X=forward, Y=sideways, Z=up)
# ─────────────────────────────────────────────
target = np.array([450, 200, -350.0])

model = mujoco.MjModel.from_xml_path("assets/scene.xml")
data  = mujoco.MjData(model)

def command_joints(q: np.ndarray):
    data.ctrl[0] = np.clip(q[0],  0.0,   6.2832)
    data.ctrl[1] = np.clip(q[1],  0.0,  3.14)
    data.ctrl[2] = np.clip(q[2],  0.0,   4.0)
    data.ctrl[3] = np.clip(q[3],  0.0,   4.1)

# ─────────────────────────────────────────────
#  Solve IK
# ─────────────────────────────────────────────
print(f"Solving IK for target: {target}")
q_sol, success, error = solve_ik(target, n_restarts=10, verbose=True)

print(f"\n  Converged    : {success}")
print(f"  Final error  : {error:.4f} units")
print(f"    q1 (swivel)   = {np.degrees(q_sol[0]):8.3f} °")
print(f"    q2 (shoulder) = {np.degrees(q_sol[1]):8.3f} °")
print(f"    q3 (elbow)    = {np.degrees(q_sol[2]):8.3f} °")
print(f"    q4 (wrist)    = {np.degrees(q_sol[3]):8.3f} °")

# ─────────────────────────────────────────────
#  FK for home and solution poses
# ─────────────────────────────────────────────
_, joint_poses_home = forward_kinematics(np.zeros(4))
_, joint_poses_sol  = forward_kinematics(q_sol)

# ─────────────────────────────────────────────
#  Plot both (non-blocking)
# ─────────────────────────────────────────────
plot_arm_3d(joint_poses_home, title="Home Position (q=[0, 0, 0, 0])")
plot_arm_3d(joint_poses_sol, target=target, title=f"IK Solution | error={error:.4f}")
plt.show(block=True)



with mujoco.viewer.launch_passive(model, data) as viewer:


    while viewer.is_running():
        command_joints(q_sol)
        mujoco.mj_step(model, data)
        viewer.sync()
