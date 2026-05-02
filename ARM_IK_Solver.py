"""
Robotic Arm Inverse Kinematics Solver — Full 3D
=================================================
4-DOF arm with Z-up world frame convention.

Coordinate Frame:
  X = forward
  Y = sideways
  Z = up (vertical)

Joint Layout:
  Joint 1 : swivel — rotates around Z axis (points arm in XY direction)
  Joint 2 : shoulder — rotates in the arm's vertical plane
  Joint 3 : elbow
  Joint 4 : wrist

Physical Parameters (from CAD):
  L1        = 402.5    (joint 2 → joint 3)
  L2        = 399.836  (joint 3 → joint 4)
  L3        = 225.075  (joint 4 → end effector)
  BEND_J3   = 66.935°  fixed mechanical bend at joint 3
  BEND_J4   = 51.982°  fixed mechanical bend at joint 4

Home Position (from CAD, q=[0,0,0,0]):
  Joint 1 = 0°        (pointing along X)
  Joint 2 = 0°        (horizontal)
  Joint 3 = 171.942°  (folded back over link 1)
  Joint 4 = -159.568° (bent down)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ─────────────────────────────────────────────
#  Robot Parameters
# ─────────────────────────────────────────────
L1 = 402.5    # link 1 length (joint 2 → joint 3)
L2 = 399.836  # link 2 length (joint 3 → joint 4)
L3 = 225.075  # link 3 length (joint 4 → end effector)

BEND_J3 = np.radians(66.935)   # fixed mechanical bend at joint 3
BEND_J4 = np.radians(51.982)   # fixed mechanical bend at joint 4

# Home offsets from CAD — geometric angles already include the bend,
# so subtract the bend out so it isn't double-counted when FK adds it back.
HOME_OFFSET_J1 = np.radians(0.0)
HOME_OFFSET_J2 = np.radians(0.0)
HOME_OFFSET_J3 = np.radians(171.942) - BEND_J3
HOME_OFFSET_J4 = np.radians(-159.568) - BEND_J4

# Height of joint 2 (shoulder) above joint 1 (swivel) along Z.
SHOULDER_HEIGHT = 207.29874

# Joint limits (radians) — adjust to match your hardware
JOINT_LIMITS = [
    (-np.pi,  np.pi),   # q1: swivel   ±180°
    (0,  np.pi),   # q2: shoulder
    (0,  4),   # q3: elbow
    (0, 4.1),   # q4: wrist
]


# ─────────────────────────────────────────────
#  DH Transformation Matrix
# ─────────────────────────────────────────────

def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Standard DH 4x4 homogeneous transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,  -st * ca,   st * sa,  a * ct],
        [st,   ct * ca,  -ct * sa,  a * st],
        [0,    sa,        ca,       d     ],
        [0,    0,         0,        1     ]
    ])

def analytical_seed(target_pos: np.ndarray) -> np.ndarray:
    tx, ty, tz = target_pos

    # q1: point swivel directly at target in XY plane
    q1 = np.arctan2(ty, tx)

    reach_xy = np.sqrt(tx**2 + ty**2)
    reach_z  = tz - SHOULDER_HEIGHT
    below_shoulder = tz < SHOULDER_HEIGHT

    # q2: elevation angle — negative when target is below shoulder
    q2 = np.arctan2(reach_z, reach_xy)
    if below_shoulder:
        # Force shoulder to fold other way
        q2 = np.clip(np.pi + np.arctan2(reach_z, reach_xy), 0, np.pi)

    # Straight-line distance from shoulder to target
    D = np.sqrt(reach_xy**2 + reach_z**2)
    D_max = L1 + L2 + L3
    D_min = abs(L1 - L2 - L3)
    D = np.clip(D, D_min + 1e-6, D_max - 1e-6)

    # q3: law of cosines
    D2 = np.clip(D, abs(L1 - L2) + 1e-6, L1 + L2 - 1e-6)
    cos_elbow = (L1**2 + L2**2 - D2**2) / (2 * L1 * L2)
    elbow_angle = np.arccos(np.clip(cos_elbow, -1.0, 1.0))

    if below_shoulder:
        # Elbow needs to fold the other way for downward reach
        q3 = elbow_angle  # don't invert for downward targets
    else:
        q3 = 2 * np.pi - elbow_angle  # standard inverted convention

    # q4: neutral wrist
    q4 = 0.0

    q = np.array([q1, q2, q3, q4])
    for i, (lo, hi) in enumerate(JOINT_LIMITS):
        q[i] = np.clip(q[i], lo, hi)

    return q
def numerical_rotation_jacobian(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    3×4 rotational Jacobian for the end effector Y axis.
    Each column: how does the EE Y axis (rot[:, 1]) change per joint?
    """
    Jr = np.zeros((3, len(q)))
    for i in range(len(q)):
        q_fwd = q.copy(); q_fwd[i] += eps
        q_bwd = q.copy(); q_bwd[i] -= eps
        R_fwd = forward_kinematics(q_fwd)[0][:3, :3]
        R_bwd = forward_kinematics(q_bwd)[0][:3, :3]

        Jr[:, i] = (R_fwd[:, 1] - R_bwd[:, 1]) / (2 * eps)
    return Jr
# ─────────────────────────────────────────────
#  Forward Kinematics
# ─────────────────────────────────────────────

def forward_kinematics(q: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Full 3D FK for 4-DOF arm (Z-up frame).

    Args:
        q : [q1, q2, q3, q4] motor-commanded joint angles (radians)
              q1 = swivel (around Z axis)
              q2 = shoulder
              q3 = elbow
              q4 = wrist

    Returns:
        T_ee        : 4x4 end-effector transform in world frame
        joint_poses : list of 4x4 transforms [base, j1, j2, j3, ee]
    """
    q1, q2, q3, q4 = q
    q3_inverted = 2 * np.pi - q3
    # Joint 1: swivel around Z axis.


    T_swivel = dh_transform(q1 + HOME_OFFSET_J1, SHOULDER_HEIGHT, 0, np.radians(90))

    # Joints 2-4: planar arm in the vertical plane
    T_j2 = dh_transform(q2 + HOME_OFFSET_J2,                0, L1, 0)
    T_j3 = dh_transform(q3_inverted + HOME_OFFSET_J3 + BEND_J3,      0, L2, 0)
    T_j4 = dh_transform(q4 + HOME_OFFSET_J4 + BEND_J4,      0, L3, 0)

    # Accumulate from world origin
    T_base  = np.eye(4)
    T_at_j1 = T_base  @ T_swivel
    T_at_j2 = T_at_j1 @ T_j2
    T_at_j3 = T_at_j2 @ T_j3
    T_ee    = T_at_j3 @ T_j4

    return T_ee, [T_base, T_at_j1, T_at_j2, T_at_j3, T_ee]


# ─────────────────────────────────────────────
#  Numerical Jacobian
# ─────────────────────────────────────────────

def numerical_jacobian(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """3×4 translational Jacobian via central finite differences."""
    J = np.zeros((3, len(q)))
    for i in range(len(q)):
        q_fwd = q.copy(); q_fwd[i] += eps
        q_bwd = q.copy(); q_bwd[i] -= eps
        p_fwd = forward_kinematics(q_fwd)[0][:3, 3]
        p_bwd = forward_kinematics(q_bwd)[0][:3, 3]
        J[:, i] = (p_fwd - p_bwd) / (2 * eps)
    return J


# ─────────────────────────────────────────────
#  Inverse Kinematics
# ─────────────────────────────────────────────

def inverse_kinematics(
    target_pos : np.ndarray,
    q_init     : np.ndarray | None = None,
    max_iter   : int   = 2000,
    tol        : float = 1e-3,
    step_size  : float = 0.5,
    damping    : float = 0.01,
    verbose    : bool  = True,
) -> tuple[np.ndarray, bool, float]:
    """
    Damped Least Squares Jacobian IK for full 4-DOF arm.

    Returns motor-commanded angles [q1, q2, q3, q4] — bend offsets and
    home offsets are handled transparently inside FK.

    Args:
        target_pos : desired EE position [x, y, z] in world frame
        q_init     : initial guess; random if None
        max_iter   : iteration cap
        tol        : convergence tolerance (same units as link lengths)
        step_size  : gradient step scale
        damping    : DLS damping factor (prevents singularity blow-up)
        verbose    : print convergence info
    """
    if q_init is None:
        q_init = analytical_seed(target_pos)

    q = np.array(q_init, dtype=float)

    for iteration in range(max_iter):
        T_ee, _ = forward_kinematics(q)
        current_pos = T_ee[:3, 3]

        error_vec = target_pos - current_pos
        error_mag = np.linalg.norm(error_vec)

        if error_mag < tol:
            if verbose:
                print(f"  ✓ Converged in {iteration} iterations | error = {error_mag:.4f}")
            return q, True, error_mag

        # Damped Least Squares: J† = Jᵀ (J Jᵀ + λ²I)⁻¹
        J = numerical_jacobian(q)
        JJT = J @ J.T
        J_dls = J.T @ np.linalg.inv(JJT + damping ** 2 * np.eye(3))

        q = q + step_size * (J_dls @ error_vec)

        for i, (lo, hi) in enumerate(JOINT_LIMITS):
            q[i] = np.clip(q[i], lo, hi)

    final_error = np.linalg.norm(target_pos - forward_kinematics(q)[0][:3, 3])
    if verbose:
        print(f"  ✗ Did not converge after {max_iter} iters | error = {final_error:.4f}")
    return q, False, final_error


def solve_ik(
    target_pos : np.ndarray,
    n_restarts : int = 8,
    **ik_kwargs,
) -> tuple[np.ndarray, bool, float]:

    ik_kwargs.setdefault("verbose", False)

    best_q, best_success, best_error = None, False, np.inf
    below_shoulder = target_pos[2] < SHOULDER_HEIGHT

    # Pre-compute analytical seed once
    base_seed  = analytical_seed(target_pos)
    elbow_seed = base_seed[2]  # use as reference for elbow perturbation

    for i in range(n_restarts):
        q_init = base_seed.copy()

        if i > 0:
            q_init += np.random.uniform(-0.2, 0.2, 4)

        if below_shoulder:
            q_init[1] = np.clip(np.pi - (0.15 * i), 0, np.pi)
            q_init[2] = np.clip(elbow_seed + np.random.uniform(-0.3, 0.3), 0, 4)

        for j, (lo, hi) in enumerate(JOINT_LIMITS):
            q_init[j] = np.clip(q_init[j], lo, hi)

        q_sol, success, error = inverse_kinematics(target_pos, q_init, **ik_kwargs)

        if error < best_error:
            best_q, best_success, best_error = q_sol, success, error

        if success:
            break

    return best_q, best_success, best_error

# ─────────────────────────────────────────────
#  Visualisation — 3D (Z-up)
# ─────────────────────────────────────────────

def plot_arm_3d(
    joint_poses : list[np.ndarray],
    target      : np.ndarray | None = None,
    title       : str = "Robotic Arm — 3D View",
):
    """
    Plots the arm in 3D with Z-up convention.
    X=forward, Y=sideways, Z=up.
    joint_poses is the list returned by forward_kinematics().
    """
    positions = [T[:3, 3] for T in joint_poses]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # Links
    ax.plot(xs, ys, zs, color="#4cc9f0", linewidth=4)

    # Joints
    colors = ["#ffffff", "#f72585", "#7209b7", "#3a0ca3", "#4361ee"]
    labels = ["Base", "Joint 1 (swivel)", "Joint 2 (shoulder)",
              "Joint 3/4 (elbow/wrist)", "End Effector"]
    for x, y, z, c, lbl in zip(xs, ys, zs, colors, labels):
        ax.scatter(x, y, z, s=120, color=c, zorder=5, label=lbl)

    # Target marker
    if target is not None:
        ax.scatter(*target, s=250, color="#f8961e", marker="*", zorder=6,
                   label=f"Target ({target[0]:.0f}, {target[1]:.0f}, {target[2]:.0f})")

    # Axis limits centred on reach
    max_reach = L1 + L2 + L3
    ax.set_xlim(-max_reach, max_reach)
    ax.set_ylim(-max_reach, max_reach)
    ax.set_zlim(0, max_reach)

    ax.set_xlabel("X — forward", color="white", labelpad=10)
    ax.set_ylabel("Y — sideways", color="white", labelpad=10)
    ax.set_zlabel("Z — up", color="white", labelpad=10)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333")
    ax.yaxis.pane.set_edgecolor("#333")
    ax.zaxis.pane.set_edgecolor("#333")
    ax.grid(True, color="#333", linestyle="--", linewidth=0.5)
    ax.legend(facecolor="#16213e", labelcolor="white", fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)  # allow plot to render


