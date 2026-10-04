from robot_sim import RobotSim

sim = RobotSim(gui=True)
sim.drop_next = True
print(sim.pick("red_cube"))
print(sim.place("blue_zone"))
red = [o for o in sim.get_scene()["objects"] if o["name"] == "red_cube"][0]
print("red_cube:", red["position"], "in_zone:", red["in_zone"])
input("Press Enter to close...")