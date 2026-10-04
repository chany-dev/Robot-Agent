import json
import time
import uuid
import agentv2 as A
from robot_sim import RobotSim

A.sim = RobotSim(gui=False)
A.APPROVAL["on"] = False

# goal = {object: zone} expected at the end. goal=None means the agent must refuse and move nothing.
CASES = [
    dict(name="single move",       cmd="put the red cube in the blue zone",                    goal={"red_cube": "blue_zone"}),
    dict(name="single move 2",     cmd="move the green cube to the gray zone",                 goal={"green_cube": "gray_zone"}),
    dict(name="single move 3",     cmd="put the yellow cube in the blue zone",                 goal={"yellow_cube": "blue_zone"}),
    dict(name="two cubes",         cmd="put the red cube in the blue zone and the green cube in the gray zone",
                                   goal={"red_cube": "blue_zone", "green_cube": "gray_zone"}),
    dict(name="typo in zone",      cmd="put the red cube in the bluezone",                     goal={"red_cube": "blue_zone"}),
    dict(name="drop failure 1",    cmd="put the red cube in the blue zone",                    goal={"red_cube": "blue_zone"}, drop=True),
    dict(name="drop failure 2",    cmd="move the green cube to the gray zone",                 goal={"green_cube": "gray_zone"}, drop=True),
    dict(name="drop + lying tool", cmd="put the yellow cube in the blue zone",                 goal={"yellow_cube": "blue_zone"}, drop=True, toolcheck=False),
    dict(name="refuse: no object", cmd="pick up the banana",                                   goal=None),
    dict(name="refuse: no zone",   cmd="put the red cube in the purple zone",                  goal=None),
    dict(name="two cubes + drop",  cmd="put the red cube in the blue zone and the yellow cube in the gray zone",
                                   goal={"red_cube": "blue_zone", "yellow_cube": "gray_zone"}, drop=True),
]


def final_state():
    return {o["name"]: o["in_zone"] for o in A.sim.get_scene()["objects"]}


def matches(goal, state):
    expected = {name: None for name in state}
    if goal:
        expected.update(goal)
    return state == expected


results = []
for i, c in enumerate(CASES, 1):
    A.sim.reset()
    A.TOOL_CHECK["on"] = c.get("toolcheck", True)
    if c.get("drop"):
        A.sim.drop_next = True
    cfg = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}, "recursion_limit": 60}

    t0 = time.time()
    err, verdict, retries, calls = None, None, 0, 0
    try:
        out = A.app.invoke({"messages": [("user", c["cmd"])]}, cfg)
        verdict = out.get("verdict")
        retries = out.get("verify_retries", 0)
        calls = sum(len(m.tool_calls) for m in out["messages"] if getattr(m, "tool_calls", None))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:200]

    state = final_state()
    ok = matches(c["goal"], state)
    injected = bool(c.get("drop")) and not A.sim.drop_next        # did the failure actually fire?
    r = dict(n=i, name=c["name"], command=c["cmd"], success=ok, verdict=verdict,
             verifier_retries=retries, tool_calls=calls, failure_injected=injected,
             seconds=round(time.time() - t0, 1), final_state=state, error=err)
    results.append(r)
    print(f"{i:>2}. {'PASS' if ok else 'FAIL'}  {c['name']:<20} verdict={verdict}  retries={retries}  "
          f"calls={calls}  injected={injected}  {r['seconds']}s" + (f"  ERROR={err}" if err else ""))
    time.sleep(2)   # be gentle with API rate limits

A.TOOL_CHECK["on"] = True
n = len(results)
passed = sum(r["success"] for r in results)
inj = [r for r in results if r["failure_injected"]]
inj_ok = sum(r["success"] for r in inj)
refuse = [r for r, c in zip(results, CASES) if c["goal"] is None]
refuse_ok = sum(r["success"] for r in refuse)
caught = sum(1 for r in results if r["verifier_retries"] > 0 and r["success"])

print("\n========== SUMMARY ==========")
print(f"Overall task success:        {passed}/{n} ({100 * passed / n:.0f}%)")
print(f"Recovery from injected failures: {inj_ok}/{len(inj)}")
print(f"Correct refusals (impossible tasks): {refuse_ok}/{len(refuse)}")
print(f"Runs rescued by the independent verifier: {caught}")
print(f"Avg tool calls per task: {sum(r['tool_calls'] for r in results) / n:.1f}")
print(f"Avg time per task: {sum(r['seconds'] for r in results) / n:.1f}s")

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved full results to eval_results.json")