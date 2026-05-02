"""
Interactive IK Controller — X-Z Click Interface
================================================
Click on the X-Z matplotlib plot to set a target.
IK is solved once per click, then joints are held at that solution.

Controls:
  Left-click  -> solve IK and move arm to (X, Z)  [Y = 0]
  Right-click -> reset to home position
  Close plot  -> exit
"""

import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mujoco
import mujoco.viewer

from ARM_IK_Solver import solve_ik, forward_kinematics, plot_arm_3d

# ─────────────────────────────────────────────
#  Robot Parameters (must match ARM_IK_Solver)
# ─────────────────────────────────────────────
L1 = 402.5
L2 = 399.836
L3 = 225.075
SHOULDER_HEIGHT = 207.29874
MAX_REACH = (L1 + L2 + L3 + SHOULDER_HEIGHT) * 0.95
EXCLUSION_RADIUS = 150

# ─────────────────────────────────────────────
#  MuJoCo Setup
# ─────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("assets/scene.xml")
data  = mujoco.MjData(model)


def check_exclusion_zone(poses):
    """Check if any arm link except end-effector enters exclusion zone."""
    for pose in poses[:-1]:
        x, z = pose[0, 3], pose[2, 3]
        if np.hypot(x, z) < EXCLUSION_RADIUS:
            return True
    return False


def command_joints(q: np.ndarray):
    data.ctrl[0] = np.clip(q[0], 0.0,    6.2832)
    data.ctrl[1] = np.clip(q[1], 0.0,    3.14)
    data.ctrl[2] = np.clip(q[2], 0.0,    4.0)
    data.ctrl[3] = np.clip(q[3], 0.0,    4.1)


# ─────────────────────────────────────────────
#  Shared State
# ─────────────────────────────────────────────
q_current  = np.zeros(4)       # current joint command
ik_ok      = True
state_lock = threading.Lock()


# ─────────────────────────────────────────────
#  MuJoCo Viewer Thread
# ─────────────────────────────────────────────
def sim_thread():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            with state_lock:
                q = q_current.copy()
            command_joints(q)
            mujoco.mj_step(model, data)
            viewer.sync()

threading.Thread(target=sim_thread, daemon=True).start()


# ─────────────────────────────────────────────
#  Click Handler — solves IK once per click
# ─────────────────────────────────────────────
def on_click(event, ax, target_dot, status_txt, fig):
    global q_current, ik_ok

    if event.inaxes != ax or event.xdata is None:
        return

    if event.button == 3:   # right-click -> home
        with state_lock:
            q_current = np.zeros(4)
            ik_ok = True
        target_dot.set_data([], [])
        status_txt.set_text("Home position")
        fig.canvas.draw_idle()
        return

    # Clamp to reachable workspace
    x, z = event.xdata, max(event.ydata, -300.0)
    r = np.hypot(x, z)
    if r > MAX_REACH:
        x, z = x * MAX_REACH / r, z * MAX_REACH / r

    target = np.array([x, 0.0, z])

    print(f"\nTarget: X={x:.1f}  Z={z:.1f} mm")
    q_sol, success, error = solve_ik(target, n_restarts=10)

    print(f"  Converged    : {success}")
    print(f"  Final error  : {error:.4f} mm")
    print(f"  Converged    : {success}")
    print(f"  Final error  : {error:.4f} mm")
    print(f"    q1 (swivel)   = {np.degrees(q_sol[0]):8.3f} deg")
    print(f"    q2 (shoulder) = {np.degrees(q_sol[1]):8.3f} deg")
    print(f"    q3 (elbow)    = {np.degrees(q_sol[2]):8.3f} deg")
    print(f"    q4 (wrist)    = {np.degrees(q_sol[3]):8.3f} deg")

    if success:
        T_ee, poses = forward_kinematics(q_sol)
        if check_exclusion_zone(poses):
            success = False

    with state_lock:
        q_current = q_sol
        ik_ok = success

    target_dot.set_data([x], [z])
    label = f"Target  X={x:.0f}  Z={z:.0f} mm  |  {'OK' if success else 'FAIL'}  err={error:.1f}"
    status_txt.set_text(label)
    fig.canvas.draw_idle()


# ─────────────────────────────────────────────
#  Matplotlib X-Z Plot
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

ax.set_xlim(-1000, 1000)
ax.set_ylim(-300, (L1 + L2 + L3) * 1.2)
ax.set_aspect("equal")
ax.set_xlabel("X — forward (mm)", color="#8b949e", fontsize=11)
ax.set_ylabel("Z — up (mm)",      color="#8b949e", fontsize=11)
ax.set_title("Left-click -> move arm    Right-click -> home",
             color="#c9d1d9", fontsize=11, pad=10)
ax.tick_params(colors="#8b949e")
for sp in ax.spines.values():
    sp.set_edgecolor("#30363d")
ax.grid(True, color="#1c2128", linewidth=0.6)
ax.axhline(0,                color="#30363d", linewidth=1.0)
ax.axvline(0,                color="#30363d", linewidth=1.0)
ax.axhline(SHOULDER_HEIGHT,  color="#3d5a80", linewidth=0.9,
           linestyle="--", alpha=0.7)
ax.text((L1 + L2 + L3) * 0.45, SHOULDER_HEIGHT + 12,
        "shoulder height", color="#3d5a80", fontsize=7, alpha=0.8)

ws = plt.Circle((0, 0), MAX_REACH, color="#388bfd",
                fill=False, linewidth=1.0, linestyle="--", alpha=0.35)
ax.add_patch(ws)

ez = plt.Circle((0, 0), EXCLUSION_RADIUS, color="#f85149",
                fill=False, linewidth=1.5, linestyle="-", alpha=0.6)
ax.add_patch(ez)

# Artists
arm_line,   = ax.plot([], [], "o-", color="#58a6ff", linewidth=2.5,
                      markersize=7,  zorder=5, label="Arm (FK)")
ee_dot,     = ax.plot([], [], "^",  color="#ffa657", markersize=9,
                      zorder=7, label="End-effector")
target_dot, = ax.plot([], [], "o",  color="#f85149", markersize=12,
                      zorder=6, label="Target")
ik_badge    = ax.text(0.98, 0.96, "", transform=ax.transAxes,
                      ha="right", va="top", fontsize=9,
                      color="#3fb950", fontfamily="monospace")
status_txt  = ax.text(0.02, 0.97, "Click to begin",
                      transform=ax.transAxes, color="#8b949e",
                      fontsize=8, va="top")
ax.legend(facecolor="#161b22", edgecolor="#30363d",
          labelcolor="#c9d1d9", loc="lower right", fontsize=8)

fig.canvas.mpl_connect(
    "button_press_event",
    lambda e: on_click(e, ax, target_dot, status_txt, fig)
)


# ─────────────────────────────────────────────
#  Animation — redraws arm overlay at 20 fps
# ─────────────────────────────────────────────
def update(_):
    with state_lock:
        q  = q_current.copy()
        ok = ik_ok

    T_ee, poses = forward_kinematics(q)
    xs = [T[0, 3] for T in poses]
    zs = [T[2, 3] for T in poses]
    arm_line.set_data(xs, zs)
    ee_dot.set_data([T_ee[0, 3]], [T_ee[2, 3]])
    collision = check_exclusion_zone(poses)
    ok = ok and not collision
    ik_badge.set_text("IK OK" if ok else "IK FAIL")
    ik_badge.set_color("#3fb950" if ok else "#f85149")
    return arm_line, ee_dot, ik_badge


ani = animation.FuncAnimation(fig, update, interval=50, blit=True)

plt.tight_layout()
plt.show()