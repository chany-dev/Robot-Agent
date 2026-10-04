from robot_sim import RobotSim

sim = RobotSim(gui=True)
print(sim.get_scene())
print(sim.pick("red_cube"))
print(sim.place("blue_zone"))
print(sim.get_scene())
print(sim.pick("banana"))   # should fail cleanly, not crash
input("Press Enter to close...")