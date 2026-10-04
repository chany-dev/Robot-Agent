import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
cube = p.loadURDF("cube_small.urdf", [0.5, 0, 0.03])

for _ in range(720):
    p.stepSimulation()
    time.sleep(1 / 240)

print("Joints:", p.getNumJoints(robot))
print("Cube position:", p.getBasePositionAndOrientation(cube)[0])
p.disconnect()