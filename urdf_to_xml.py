import mujoco
model = mujoco.MjModel.from_xml_path("ARM/assets/robot.urdf")
mujoco.mj_saveLastXML("robot.xml", model)