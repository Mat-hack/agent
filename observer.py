from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from state import GitState

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)


OBSERVER_PROMPT = SystemMessage(
    content="""
You are a Git Workflow Reviewer.

You receive:

- Current TODO
- Command executed
- Command output

Your job is ONLY to determine whether the CURRENT TODO
was completed by the command that just ran.

IMPORTANT

- Judge only the current command execution.
- Do not infer user intent.
- Do not assume existing state satisfies a TODO.
- Do not mark a TODO complete merely because something already exists.
- A TODO is complete only if the command execution successfully achieved it.

Examples:

TODO:
Create branch potter

OUTPUT:
Switched to a new branch 'potter'

Response:

TODO_COMPLETE: YES

ANALYSIS:
Branch was created successfully.


TODO:
Create branch potter

OUTPUT:
fatal: a branch named 'potter' already exists

Response:

TODO_COMPLETE: NO

ANALYSIS:
Branch already exists. The requested creation did not occur.


TODO:
Find latest commit hash on developer

OUTPUT:
80b774b62f966399949256de9c52c535822a6a1d

Response:

TODO_COMPLETE: YES

ANALYSIS:
The requested hash was obtained.
"""
)

def observer_node(state: GitState):

    todos = state.get("todos", [])
    current_idx = state.get("current_todo", 0)

    if current_idx >= len(todos):
        return {
            "done": True
        }

    current_todo = todos[current_idx]

    latest_message = ""

    if state.get("messages"):
        latest_message = state["messages"][-1].content

    human_response = state.get("human_response", "")
    correction_note = ""
    if human_response:
        correction_note = f"""
HUMAN CORRECTION:
The human overrode a detail of this todo with: "{human_response}".
Judge completion against their corrected intent, not the literal original
wording (e.g. if the todo says "branch X" but the human said "use Y
instead", a successful action on Y satisfies this todo).
"""

    observation_input = f"""
CURRENT TODO:
{current_todo}
{correction_note}
{latest_message}
"""

    response = llm.invoke(
        [
            OBSERVER_PROMPT,
            HumanMessage(content=observation_input)
        ]
    )

    review = response.content.strip()

    print("\n===== OBSERVER =====")
    print(f"TODO: {current_todo}")
    print(review)

    todo_complete = (
        "TODO_COMPLETE: YES" in review.upper()
    )

    return {
        "review": review,
        "todo_complete": todo_complete,
        "done": False,
        # Consumed: the next todo (if any) must not see this as "the human
        # just answered" — it didn't ask anything.
        "human_response": "",
    }

def observer_router(state: GitState):
    return "coordinator"