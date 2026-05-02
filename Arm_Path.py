import numpy as np
import tkinter as tk
import mujoco
import mujoco.viewer

from ARM_IK_Solver import solve_ik, forward_kinematics, inverse_kinematics

# Global arrays
path_points   = None
joint_path    = None
path_failures = None

# ─────────────────────────────────────────────
#  Define the Circle Path
# ─────────────────────────────────────────────
def generate_circle(center, radius, n_points=50, angle_xy=0):
    """
    Generate a vertical circle centered at a specified point.
    
    Args:
        center: [x, y, z] center of the circle
        radius: radius of the circle
        n_points: number of points on the circle
        angle_xy: angle in XY plane (radians) defining circle's orientation.
                  0 = circle in YZ plane, π/4 = rotated 45°, etc.
    """
    center = np.asarray(center, dtype=float)
    
    # Direction in XY plane (normal to the circle's plane)
    normal_xy = np.array([np.cos(angle_xy), np.sin(angle_xy), 0])
    
    # Two perpendicular vectors spanning the circle's plane
    # v1: vertical direction (Z axis)
    v1 = np.array([0, 0, 1])
    # v2: perpendicular to both normal_xy and v1
    v2 = np.cross(normal_xy, v1)
    
    # Generate circle points
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    points = np.array([
        center + radius * (np.cos(a) * v1 + np.sin(a) * v2)
        for a in angles
    ])
    return points

# ─────────────────────────────────────────────
#  PathWindow class (unchanged)
# ─────────────────────────────────────────────
class PathWindow:
    def __init__(self, points, center, width=600, height=600, margin=40):
        self.points = np.asarray(points)
        self.center = np.asarray(center)
        self.width = width
        self.height = height
        self.margin = margin
        self.closed = False

        self.root = tk.Tk()
        self.root.title("Path Viewer")
        self.canvas = tk.Canvas(
            self.root, width=self.width, height=self.height,
            bg="#111111", highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self._compute_bounds()
        self._draw_static()
        self.current_id = self.canvas.create_oval(0, 0, 0, 0, fill="#f8961e", outline="")
        self.text_id = self.canvas.create_text(
            self.width - self.margin, self.margin, anchor="ne",
            fill="white", font=("Segoe UI", 10, "bold"), text="",
        )
        self.update_current(self.points[0])
        self.safe_update()

    def _compute_bounds(self):
        ys, zs = self.points[:, 1], self.points[:, 2]
        self.min_y, self.max_y = float(np.min(ys)), float(np.max(ys))
        self.min_z, self.max_z = float(np.min(zs)), float(np.max(zs))
        if self.min_y == self.max_y: self.min_y -= 1.0; self.max_y += 1.0
        if self.min_z == self.max_z: self.min_z -= 1.0; self.max_z += 1.0

    def _world_to_canvas(self, points):
        """Vectorized conversion from world coordinates to canvas coordinates."""
        if points.ndim == 1:
            points = points.reshape(1, -1)
        ys, zs = points[:, 1], points[:, 2]
        tx = self.margin + (ys - self.min_y) * (self.width - 2 * self.margin) / (self.max_y - self.min_y)
        ty = self.height - self.margin - (zs - self.min_z) * (self.height - 2 * self.margin) / (self.max_z - self.min_z)
        return np.column_stack([tx, ty])

    def _draw_static(self):
        coords = self._world_to_canvas(self.points).flatten().tolist()
        self.canvas.create_line(*coords, fill="#f8961e", width=2, smooth=True)
        cx, cy = self._world_to_canvas(self.center).flatten()
        self.canvas.create_line(cx - 8, cy, cx + 8, cy, fill="#ffffff", width=2)
        self.canvas.create_line(cx, cy - 8, cx, cy + 8, fill="#ffffff", width=2)
        self.canvas.create_text(
            self.margin, self.margin, anchor="nw", fill="#ffffff",
            font=("Segoe UI", 9), text="Y-Z path projection",
        )

    def update_current(self, point, index=None):
        if self.closed: return
        cx, cy = self._world_to_canvas(point).flatten()
        r = 6
        self.canvas.coords(self.current_id, cx - r, cy - r, cx + r, cy + r)
        if index is not None:
            self.canvas.itemconfigure(self.text_id, text=f"waypoint {index}")

    def safe_update(self):
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def close(self):
        if not self.closed:
            self.root.destroy()
            self.closed = True

# ─────────────────────────────────────────────
#  Pre-solve IK for every waypoint
# ─────────────────────────────────────────────
def solve_path(waypoints, n_restarts=6, verbose=False):
    q_path   = np.zeros((len(waypoints), 4))
    failures = []
    q_prev   = None

    for i, wp in enumerate(waypoints):
        if q_prev is None:
            q_sol, success, error = solve_ik(wp, n_restarts=n_restarts)
        else:
            q_sol, success, error = inverse_kinematics(wp, q_init=q_prev, verbose=False)

        if not success:
            failures.append(i)
            if verbose:
                print(f"  ⚠ Waypoint {i} did not converge | error={error:.4f}")

        q_path[i] = q_sol
        q_prev    = q_sol

        if verbose:
            print(f"  [{i+1}/{len(waypoints)}] error={error:.4f}")

    return q_path, failures


def main(display_path=True):
    global path_points, joint_path, path_failures  

    center   = np.array([650, 0, 700])  # raised Z so full circle is reachable
    radius   = 150.0
    n_points = 200  # ← defined here, so everything below can use it

    print("Generating circle path...")
    waypoints = generate_circle(center, radius, n_points)

    print(f"Solving IK for {n_points} waypoints...")
    q_path, failures = solve_path(waypoints, verbose=True)
    print(f"Done. Failures: {len(failures)}/{n_points}")

    # ── Verify actual EE positions vs requested waypoints ──────
    actual_positions = np.array([forward_kinematics(q)[0][:3, 3] for q in q_path], dtype=np.float32)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")
    
    # Configure both axes at once
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.set_aspect("equal")

    axes[0].plot(waypoints[:, 0], waypoints[:, 2], color="#f8961e", label="Requested", linewidth=2)
    axes[0].plot(actual_positions[:, 0], actual_positions[:, 2], color="#4cc9f0", label="Actual EE", linewidth=2, linestyle="--")
    axes[0].set_xlabel("X", color="white")
    axes[0].set_ylabel("Z", color="white")
    axes[0].set_title("XZ plane", color="white")
    axes[0].legend(labelcolor="white", facecolor="#16213e")

    axes[1].plot(waypoints[:, 0], waypoints[:, 1], color="#f8961e", label="Requested", linewidth=2)
    axes[1].plot(actual_positions[:, 0], actual_positions[:, 1], color="#4cc9f0", label="Actual EE", linewidth=2, linestyle="--")
    axes[1].set_xlabel("X", color="white")
    axes[1].set_ylabel("Y", color="white")
    axes[1].set_title("XY plane", color="white")
    axes[1].legend(labelcolor="white", facecolor="#16213e")

    plt.suptitle("Requested vs Actual EE Path", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    # Store results globally
    path_points   = waypoints
    joint_path    = q_path
    path_failures = failures

    path_window = None
    if display_path:
        path_window = PathWindow(waypoints, center)

    model = mujoco.MjModel.from_xml_path("assets/scene.xml")
    data  = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)

    def command_joints(q: np.ndarray):
        data.ctrl[0] = np.clip(q[0], -3.1415, 3.1415)
        data.ctrl[1] = np.clip(q[1], 0.0, 3.14)
        data.ctrl[2] = np.clip(q[2], 0.0, 4.0)
        data.ctrl[3] = np.clip(q[3], 0.0, 4.1)

    STEPS_PER_WAYPOINT = 20  # how many simulation steps to hold each waypoint before moving to the next

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_counter = 0
        has_window = path_window is not None
        while viewer.is_running():
            if step_counter % STEPS_PER_WAYPOINT == 0:
                waypoint_idx = (step_counter // STEPS_PER_WAYPOINT) % n_points
                command_joints(q_path[waypoint_idx])
                if has_window and not path_window.closed:
                    path_window.update_current(waypoints[waypoint_idx], index=waypoint_idx)

            mujoco.mj_step(model, data)
            viewer.sync()
            if has_window and not path_window.closed:
                path_window.safe_update()
            step_counter += 1

    if path_window is not None and not path_window.closed:
        path_window.close()

    return waypoints, q_path, failures


if __name__ == "__main__":
    main(display_path=True)