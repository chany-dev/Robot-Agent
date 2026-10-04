import os
from typing import TypedDict, List, Optional, Literal
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from robot_sim import RobotSim

load_dotenv()
MAX_ATTEMPTS = 4
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
llm = ChatGroq(model=MODEL, temperature=0)


class Step(BaseModel):
    action: Literal["pick", "place"]
    target: str


class Plan(BaseModel):
    steps: List[Step]
    explanation: str


class State(TypedDict):
    command: str
    scene: dict
    plan: Optional[dict]
    errors: List[str]
    results: List[dict]
    attempts: int
    status: str


def observe(state: State):
    scene = sim.get_scene()
    print(f"[observe] {len(scene['objects'])} objects, holding={scene['holding']}")
    return {"scene": scene}


def planner(state: State):
    feedback = ""
    if state["errors"]:
        feedback = ("\n\nProblems with your previous attempt:\n- "
                    + "\n- ".join(state["errors"])
                    + "\nThe scene above is the CURRENT state. Make a new plan that fixes these.")
    prompt = f"""You are a task planner for a robot arm. Output a plan using ONLY these actions:
- pick(target=<object name>): pick up one object. The gripper holds one object at a time.
- place(target=<zone name>): place the held object into a zone.

Rules:
- Use only object and zone names that appear in the scene.
- Place the held object before picking another one.
- Each zone holds only ONE cube, so use different zones for different cubes.
- If the command is impossible or refers to things not in the scene, return empty steps and explain why.
- Only plan what is still needed: if an object is already in the right zone, skip it.

Scene (JSON): {state['scene']}
Command: {state['command']}{feedback}"""
    structured = llm.with_structured_output(Plan, method="function_calling")
    plan = structured.invoke(prompt)
    print(f"[planner] attempt {state['attempts'] + 1}: {[s.model_dump() for s in plan.steps]}")
    print(f"[planner] reason: {plan.explanation}")
    return {"plan": plan.model_dump(), "errors": [], "attempts": state["attempts"] + 1}


def validator(state: State):
    scene, steps = state["scene"], state["plan"]["steps"]
    if not steps:
        print("[validator] empty plan -> refused")
        return {"status": "refused"}

    objs = {o["name"]: o for o in scene["objects"]}
    zones = scene["zones"]
    xr, yr = scene["workspace"]["x"], scene["workspace"]["y"]
    holding, errors = scene["holding"], []

    for i, s in enumerate(steps, 1):
        a, t = s["action"], s["target"]
        if a == "pick":
            if t not in objs:
                errors.append(f"step {i}: unknown object '{t}'. Valid: {list(objs)}")
            elif holding:
                errors.append(f"step {i}: cannot pick '{t}' while holding '{holding}'")
            else:
                x, y, _ = objs[t]["position"]
                if not (xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1]):
                    errors.append(f"step {i}: '{t}' is outside the workspace")
                else:
                    holding = t
        else:
            if t not in zones:
                errors.append(f"step {i}: unknown zone '{t}'. Valid: {list(zones)}")
            elif not holding:
                errors.append(f"step {i}: cannot place, gripper is empty")
            else:
                holding = None

    print(f"[validator] {'OK' if not errors else errors}")
    return {"errors": errors, "status": "rejected" if errors else "approved"}


def executor(state: State):
    results = []
    for s in state["plan"]["steps"]:
        r = getattr(sim, s["action"])(s["target"])
        print(f"[executor] {s['action']}({s['target']}) -> {r}")
        results.append({"step": s, **r})
        if not r["ok"]:
            break
    return {"results": results}


def verifier(state: State):
    """Checks the real world, not what the tools claimed."""
    scene = sim.get_scene()
    in_zone = {o["name"]: o["in_zone"] for o in scene["objects"]}

    expected, held = [], None
    for s in state["plan"]["steps"]:
        if s["action"] == "pick":
            held = s["target"]
        else:
            expected.append((held, s["target"]))

    problems = []
    for r in state["results"]:
        if not r["ok"]:
            problems.append(f"{r['step']['action']}({r['step']['target']}) failed: {r['message']}")
    for obj, zone in expected:
        if in_zone.get(obj) != zone:
            problems.append(f"after execution, '{obj}' is NOT in '{zone}' (it is in: {in_zone.get(obj)})")

    if problems:
        print(f"[verifier] FAIL -> {problems}")
        return {"errors": problems, "status": "failed"}
    print("[verifier] PASS - world matches the plan")
    return {"errors": [], "status": "verified"}


def after_validator(state: State):
    if state["status"] == "approved":
        return "executor"
    if state["status"] == "rejected" and state["attempts"] < MAX_ATTEMPTS:
        return "planner"
    return END


def after_verifier(state: State):
    if state["status"] == "failed" and state["attempts"] < MAX_ATTEMPTS:
        print("[replan] going back to observe and plan again")
        return "observe"
    return END


g = StateGraph(State)
g.add_node("observe", observe)
g.add_node("planner", planner)
g.add_node("validator", validator)
g.add_node("executor", executor)
g.add_node("verifier", verifier)
g.add_edge(START, "observe")
g.add_edge("observe", "planner")
g.add_edge("planner", "validator")
g.add_conditional_edges("validator", after_validator, ["executor", "planner", END])
g.add_edge("executor", "verifier")
g.add_conditional_edges("verifier", after_verifier, ["observe", END])
app = g.compile()


if __name__ == "__main__":
    sim = RobotSim(gui=True)
    print(f"Using Groq model: {MODEL}")
    print("Commands: type a task, or '!drop' (inject a failure), '!reset' (reset scene), 'quit'")
    while True:
        cmd = input("\nCommand: ").strip()
        if cmd.lower() in ("quit", "exit", ""):
            break
        if cmd == "!drop":
            sim.drop_next = True
            print("Failure armed: the cube will slip during the next place.")
            continue
        if cmd == "!reset":
            sim.reset()
            print("Scene reset.")
            continue
        out = app.invoke({"command": cmd, "scene": {}, "plan": None, "errors": [],
                          "results": [], "attempts": 0, "status": ""})
        print(f"==> final status: {out['status']} (attempts: {out['attempts']})")