import math
import time
import pybullet as p
import pybullet_data


class RobotSim:
    EE = 11                      # Panda end-effector link
    ARM = list(range(7))         # arm joints
    FINGERS = [9, 10]            # gripper joints
    HOME = [0, -0.3, 0, -2.2, 0, 2.0, 0.785]
    X_RANGE = (0.25, 0.75)       # reachable workspace (metres)
    Y_RANGE = (-0.45, 0.45)
    HOVER = 0.2                  # safe height above targets
    CUBE_H = 0.025               # half-height of a cube

    def __init__(self, gui=True):
        self.gui = gui
        if gui:
            p.connect(p.GUI, options="--width=1280 --height=720")
        else:
            p.connect(p.DIRECT)
        self.reset()

    # ---------- setup ----------
    def reset(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        for i, q in zip(self.ARM, self.HOME):
            p.resetJointState(self.robot, i, q)
        for j in self.FINGERS:
            p.resetJointState(self.robot, j, 0.04)

        self.objects, self.zones = {}, {}
        self.holding, self.constraint = None, None
        self.drop_next = False

        self._add_cube("red_cube", [0.5, 0.0], [1, 0, 0, 1])
        self._add_cube("green_cube", [0.4, 0.15], [0, 0.8, 0, 1])
        self._add_cube("yellow_cube", [0.4, -0.15], [1, 0.9, 0, 1])
        self._add_zone("blue_zone", [0.55, -0.3], [0, 0, 1, 1])
        self._add_zone("gray_zone", [0.55, 0.3], [0.6, 0.6, 0.6, 1])
        self._step(100)
        self._setup_view()

    def _setup_view(self):
        if not self.gui:
            return
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)               # hides all side panels
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)     # no accidental dragging of objects
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=50,
                                     cameraPitch=-35, cameraTargetPosition=[0.45, 0, 0.1])

    def _add_cube(self, name, xy, color):
        h = self.CUBE_H
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[h, h, h])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[h, h, h], rgbaColor=color)
        self.objects[name] = p.createMultiBody(0.1, col, vis, [xy[0], xy[1], h + 0.001])

    def _add_zone(self, name, xy, color):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.002], rgbaColor=color)
        p.createMultiBody(0, -1, vis, [xy[0], xy[1], 0.002])
        self.zones[name] = xy

    # ---------- low-level helpers ----------
    def _step(self, n=1):
        for _ in range(n):
            p.stepSimulation()
            if self.gui:
                time.sleep(1 / 240)

    def _ee_pos(self):
        return p.getLinkState(self.robot, self.EE)[4]

    def _dist(self, target):
        return math.dist(self._ee_pos(), target)

    def _obj_pos(self, name):
        return p.getBasePositionAndOrientation(self.objects[name])[0]

    def _reachable(self, x, y):
        return self.X_RANGE[0] <= x <= self.X_RANGE[1] and self.Y_RANGE[0] <= y <= self.Y_RANGE[1]

    def _res(self, ok, msg):
        return {"ok": ok, "message": msg}

    def _move_ee(self, pos, max_steps=1500, tol=0.015):
        orn = p.getQuaternionFromEuler([math.pi, 0, 0])   # gripper pointing down
        q = p.calculateInverseKinematics(self.robot, self.EE, pos, orn, maxNumIterations=100)
        for i in self.ARM:
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, q[i],
                                    force=300, maxVelocity=1.5)
        for _ in range(max_steps):
            self._step(1)
            if self._dist(pos) < tol:
                return True
        return False

    def _gripper(self, opening):
        for j in self.FINGERS:
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, opening, force=30)
        self._step(60)

    # ---------- TOOLS (the LLM will call only these) ----------
    def get_scene(self):
        objs = []
        for name in self.objects:
            pos = self._obj_pos(name)
            in_zone = next((z for z, (zx, zy) in self.zones.items()
                            if abs(pos[0] - zx) < 0.08 and abs(pos[1] - zy) < 0.08), None)
            objs.append({"name": name,
                         "position": [round(v, 3) for v in pos],
                         "in_zone": in_zone})
        return {"objects": objs,
                "zones": self.zones,
                "holding": self.holding,
                "workspace": {"x": self.X_RANGE, "y": self.Y_RANGE}}

    def go_home(self):
        for i, q in zip(self.ARM, self.HOME):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, q,
                                    force=300, maxVelocity=1.5)
        self._step(300)
        return self._res(True, "moved to home pose")

    def move_to(self, x, y, z):
        if not self._reachable(x, y) or not (0.02 <= z <= 0.6):
            return self._res(False, f"target ({x}, {y}, {z}) is outside the workspace")
        ok = self._move_ee([x, y, z])
        return self._res(ok, "moved" if ok else "could not reach target")

    def pick(self, name):
        if self.holding:
            return self._res(False, f"already holding {self.holding}")
        if name not in self.objects:
            return self._res(False, f"unknown object '{name}'")
        x, y, z = self._obj_pos(name)
        if not self._reachable(x, y):
            return self._res(False, f"'{name}' is outside the workspace")
        self._gripper(0.04)
        if not self._move_ee([x, y, z + self.HOVER]) or not self._move_ee([x, y, z]):
            return self._res(False, f"could not reach '{name}'")
        self._gripper(0.0)
        self.constraint = p.createConstraint(self.robot, self.EE, self.objects[name], -1,
                                             p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
        self.holding = name
        self._move_ee([x, y, z + self.HOVER])
        return self._res(True, f"picked {name}")

    def place(self, zone):
        if not self.holding:
            return self._res(False, "not holding anything")
        if zone not in self.zones:
            return self._res(False, f"unknown zone '{zone}'")
        x, y = self.zones[zone]
        if not self._reachable(x, y):
            return self._res(False, f"zone '{zone}' is outside the workspace")

        if getattr(self, "drop_next", False):      # injected failure
            self.drop_next = False
            print("[sim] FAILURE INJECTED: cube slipped from the gripper")
            if self.constraint is not None:
                p.removeConstraint(self.constraint)
                self.constraint = None
            self._gripper(0.04)                    # open fingers so the cube really falls
            self._step(120)

        self._move_ee([x, y, self.CUBE_H + self.HOVER])
        self._move_ee([x, y, self.CUBE_H + 0.01])
        if self.constraint is not None:
            p.removeConstraint(self.constraint)
            self.constraint = None
        name, self.holding = self.holding, None
        self._gripper(0.04)
        self._move_ee([x, y, self.CUBE_H + self.HOVER])
        self._step(60)
        return self._res(True, f"placed {name} in {zone}")

