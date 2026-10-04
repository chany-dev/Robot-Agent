import os
import json
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from robot_sim import RobotSim

load_dotenv()
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
sim = None
APPROVAL = {"on": True}
TOOL_CHECK = {"on": True}      # turn off to simulate a tool that lies about success


# ---------- tools ----------
@tool
def get_scene() -> str:
    """Look at the workspace. Returns every object's position and which zone it is in,
    the available zones, and what the gripper is holding. Call this before acting
    and again after acting to confirm the result."""
    return json.dumps(sim.get_scene())


@tool
def pick_object(name: str) -> str:
    """Pick up one object by its exact name (for example 'red_cube').
    The gripper holds one object at a time."""
    return json.dumps(sim.pick(name))


@tool
def place_object(zone: str) -> str:
    """Place the object currently held into a zone by its exact name (for example 'blue_zone').
    A zone holds only one cube. The result includes whether the object is really in the zone."""
    held = sim.holding
    result = sim.place(zone)
    if TOOL_CHECK["on"] and result["ok"] and held:
        obj = next(o for o in sim.get_scene()["objects"] if o["name"] == held)
        result["observed_zone"] = obj["in_zone"]
        result["verified"] = (obj["in_zone"] == zone)
        if not result["verified"]:
            result["ok"] = False
            result["message"] += (f" BUT the scene shows {held} is NOT in {zone} "
                                  f"(it is at {obj['position']}). The placement FAILED.")
    return json.dumps(result)


tools = [get_scene, pick_object, place_object]

SYSTEM = """You control a robot arm through tools. Work step by step.

Rules:
- Call get_scene first to see the objects, zones and gripper state. The scene can change between
  user messages, so never rely on older scenes from the conversation.
- Call ONE tool at a time and read its result before the next action.
- Use only object and zone names that exist in the scene.
- If a tool result has ok=false or verified=false, that action FAILED. Call get_scene,
  see where the object really is, and recover (for example, pick it up from where it now is).
- If a human operator rejected an action, do NOT retry it. Briefly tell the user it was
  cancelled and ask what they would like instead.
- If a message starting with [VERIFIER] says the goal is not met, call get_scene and fix it.
- Never pick up an object that is already in its target zone.
- Your LAST tool call before answering must be get_scene. Base your final answer ONLY on
  that latest scene, and never claim anything the tool outputs do not show.
- Each zone holds only one cube. If the task is impossible or refers to things that do not exist,
  say so clearly and do not act.
- When done, reply with a short summary of what you did and the verified final state."""


# ---------- state ----------
class AgentState(MessagesState):
    goal: list
    turn_start: int
    verify_retries: int
    verdict: str


class GoalItem(BaseModel):
    object: str
    zone: str


class Goal(BaseModel):
    items: List[GoalItem]
    explanation: str


base_llm = ChatGroq(model=MODEL, temperature=0)
agent_llm = base_llm.bind_tools(tools)
goal_llm = base_llm.with_structured_output(Goal, method="function_calling")


# ---------- nodes ----------
def parse_goal(state: AgentState):
    scene = sim.get_scene()
    prompt = f"""Extract the END-STATE goal of the user's command as (object, zone) pairs.
Use exact names from the scene. If the command is not a request to move objects into zones,
or refers to objects/zones that do not exist, return an empty list.

Scene: {json.dumps(scene)}
Command: {state['messages'][-1].content}"""
    try:
        goal = goal_llm.invoke(prompt)
        valid_objects = {o["name"] for o in scene["objects"]}
        valid_zones = set(scene["zones"])
        items = [i.model_dump() for i in goal.items
                 if i.object in valid_objects and i.zone in valid_zones]
    except Exception as e:
        print(f"[goal] no structured goal ({type(e).__name__}); treating as no checkable goal")
        items = []
    print(f"[goal] {items}")
    return {"goal": items, "turn_start": len(state["messages"]) - 1,
            "verify_retries": 0, "verdict": ""}


def agent(state: AgentState):
    reply = agent_llm.invoke([SystemMessage(content=SYSTEM)] + state["messages"])
    return {"messages": [reply]}


def approve(state: AgentState):
    """Human-in-the-loop: pause the graph and ask the operator before the arm moves."""
    last = state["messages"][-1]
    picks = [tc["args"].get("name", "?") for tc in last.tool_calls if tc["name"] == "pick_object"]
    answer = interrupt(f"The agent wants to pick up: {', '.join(picks)}.")
    if str(answer).strip().lower() in ("y", "yes", ""):
        return Command(goto="tools")
    rejected = [
        ToolMessage(
            content=json.dumps({"ok": False,
                                "message": f"Human operator REJECTED this action. Reason: {answer}"}),
            tool_call_id=tc["id"])
        for tc in last.tool_calls
    ]
    return Command(update={"messages": rejected}, goto="agent")


def verify(state: AgentState):
    """Independent check: plain Python compares the REAL scene with the parsed goal."""
    turn = state["messages"][state["turn_start"]:]
    if any(isinstance(m, ToolMessage) and "REJECTED" in str(m.content) for m in turn):
        print("[verifier] skipped: operator rejected an action")
        return {"verdict": "CANCELLED BY OPERATOR"}
    if not state["goal"]:
        print("[verifier] no checkable goal (refused, or not a move task)")
        return {"verdict": "NO CHECKABLE GOAL"}

    in_zone = {o["name"]: o["in_zone"] for o in sim.get_scene()["objects"]}
    problems = [f"{g['object']} should be in {g['zone']} but is in {in_zone[g['object']]}"
                for g in state["goal"] if in_zone[g["object"]] != g["zone"]]

    if not problems:
        print("[verifier] PASS - the real scene matches the goal")
        return {"verdict": "VERIFIED"}

    print(f"[verifier] FAIL -> {problems}")
    if state["verify_retries"] < 2:
        msg = HumanMessage(content="[VERIFIER] An independent check found the goal is NOT met: "
                                   + "; ".join(problems) + ". Call get_scene and fix it.")
        return {"messages": [msg], "verify_retries": state["verify_retries"] + 1, "verdict": "RETRY"}
    return {"verdict": "FAILED"}


# ---------- routing ----------
def route_agent(state: AgentState):
    last = state["messages"][-1]
    if not last.tool_calls:
        return "verify"
    if APPROVAL["on"] and any(tc["name"] == "pick_object" for tc in last.tool_calls):
        return "approve"
    return "tools"


def route_verify(state: AgentState):
    return "agent" if state["verdict"] == "RETRY" else END


g = StateGraph(AgentState)
g.add_node("parse_goal", parse_goal)
g.add_node("agent", agent)
g.add_node("approve", approve)
g.add_node("tools", ToolNode(tools))
g.add_node("verify", verify)
g.add_edge(START, "parse_goal")
g.add_edge("parse_goal", "agent")
g.add_conditional_edges("agent", route_agent, ["approve", "tools", "verify"])
g.add_edge("tools", "agent")
g.add_conditional_edges("verify", route_verify, ["agent", END])
app = g.compile(checkpointer=MemorySaver())


# ---------- running it ----------
thread = {"n": 1}


def cfg():
    return {"configurable": {"thread_id": f"session-{thread['n']}"}, "recursion_limit": 60}


def run(command: str):
    inputs = {"messages": [("user", command)]}
    while True:
        for update in app.stream(inputs, cfg(), stream_mode="updates"):
            for node, data in update.items():
                if node == "__interrupt__" or not data:
                    continue
                for m in data.get("messages", []):
                    if node == "agent":
                        for tc in (m.tool_calls or []):
                            print(f"[agent] -> {tc['name']}({tc['args']})")
                        if m.content:
                            print(f"[agent] {m.content}")
                    elif node == "tools" or node == "approve":
                        print(f"[tool]  {m.content}")
        state = app.get_state(cfg())
        if not state.next:
            print(f"==> verifier verdict: {state.values.get('verdict')}")
            break
        question = state.tasks[0].interrupts[0].value
        answer = input(f"\n[APPROVAL NEEDED] {question}\n  Press Enter / y to approve, or type a reason to reject: ")
        inputs = Command(resume=answer)


if __name__ == "__main__":
    sim = RobotSim(gui=True)
    print(f"Using Groq model: {MODEL}")
    print("Commands: a task | !drop | !reset | !approve on/off | !toolcheck on/off | quit")
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
            thread["n"] += 1
            print("Scene reset, memory cleared.")
            continue
        if cmd.startswith("!approve"):
            APPROVAL["on"] = cmd.endswith("on")
            print(f"Human approval: {'ON' if APPROVAL['on'] else 'OFF'}")
            continue
        if cmd.startswith("!toolcheck"):
            TOOL_CHECK["on"] = cmd.endswith("on")
            print(f"Tool self-check: {'ON' if TOOL_CHECK['on'] else 'OFF (tools may now report false success)'}")
            continue
        try:
            run(cmd)
        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}")