from langgraph.graph import StateGraph, END

from state import GitState

from planner import planner_node
from executor import executor_node
from observer import observer_node
from coordinator import coordinator_node
from human import human_node


def planner_router(state: GitState):
    """
    Planner can immediately finish
    for informational requests.
    """

    if state.get("done"):
        return "end"

    return "coordinator"


def entry_router(state: GitState):
    """
    main.py calls app.invoke() fresh on every turn, including every turn
    that just resumes after a human answer. Re-entering at the planner on
    those resumes would regenerate the whole plan from the growing message
    history each time, discarding progress and drifting from the original
    intent. Only go to the planner for a genuinely new/empty plan (or an
    explicit replan request); otherwise resume the in-progress plan
    directly at the coordinator, with the human's answer already in state.
    """

    if state.get("replan"):
        return "planner"

    todos = state.get("todos") or []
    current_idx = state.get("current_todo", 0)

    if todos and current_idx < len(todos):
        return "coordinator"

    return "planner"


def observer_router(state: GitState):
    """
    Observer controls workflow progress.
    """

    if state.get("done"):
        return "end"

    return "coordinator"


def coordinator_router(state: GitState):
    if state.get("done"):
        return "end"
    if state.get("waiting_for_user"):
        return "end"
    if state.get("replan"):
        return "planner"  # back to planner
    return "executor"


graph = StateGraph(GitState)

# Nodes

graph.add_node("planner", planner_node)

graph.add_node("executor", executor_node)

graph.add_node("observer", observer_node)

graph.add_node("coordinator", coordinator_node)

graph.add_node("human",human_node)

# Entry

graph.set_conditional_entry_point(
    entry_router,
    {
        "planner": "planner",
        "coordinator": "coordinator",
    }
)

# Planner routing

graph.add_conditional_edges(
    "planner",
    planner_router,
    {
        "coordinator": "coordinator",
        "end": END,
    }
)

# Command execution

graph.add_edge(
    "executor",
    "observer"
)

# Observer routing

graph.add_conditional_edges(
    "observer",
    observer_router,
    {
        "coordinator": "coordinator",
        "end": END,
    }
)

# Coordinator routing

graph.add_conditional_edges(
    "coordinator",
    coordinator_router,
    {
        "executor": "executor",
        "human": "human",
        "end": END,
    }
)
graph.add_edge(
    "human",
    "coordinator"
)

# Compile graph

app = graph.compile()