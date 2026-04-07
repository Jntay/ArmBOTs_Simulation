import mujoco
import mujoco.viewer

# Load your robot (this creates "model")
model = mujoco.MjModel.from_xml_path("robot.urdf")

# Create simulation data
data = mujoco.MjData(model)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)