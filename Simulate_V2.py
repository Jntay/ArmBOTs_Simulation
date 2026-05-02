import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("assets/scene.xml")
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)

mujoco.viewer.launch(model, data)