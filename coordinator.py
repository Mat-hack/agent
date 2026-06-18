from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from state import GitState

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

COORDINATOR_PROMPT = SystemMessage(
    content="""
You are a Git Workflow Manager.

You are given:
- Original user request
- Current TODO
- Execution history
- Previous command outputs
- Observer review


The Observer has already reviewed the latest command execution.

Your responsibilities:

1. Read and trust the Observer review.
2. Decide whether the current TODO requires:
   - another command,
   - more investigation,
   - human input,
   - or progression to the next TODO.
3. Generate exactly one shell command when appropriate.
4. Request human input when user intent is ambiguous.
5. Never perform prohibited actions.
6. If it is said commit 1 then commit 1 is the oldest commit so that's how we mark things.

CONTEXT RULE:
Before generating the next command, scan all previous command outputs for values you need.
If a previous output contains a commit hash, branch name, or file name the next TODO requires — use it directly.
Do not re-fetch information you already have.

Example:
Previous output: 80b774b62f966399949256de9c52c535822a6a1d
Next TODO: Create branch derik at that commit
Correct: git branch derik 80b774b62f966399949256de9c52c535822a6a1d
Wrong: echo "All TODOs completed"
Wrong: git rev-parse developer

IMPORTANT:
- Use the Observer review as your primary source of truth.
- If Observer indicates TODO is complete, move to next TODO.
- If Observer indicates TODO is not complete, determine safest next action.
- Prefer investigation over assumptions.
- Never guess commit hashes, branch names, or repository state.
- NEVER output echo, print, or any non-git command.
- NEVER announce completion. Just execute the next required command.

PROHIBITED ACTIONS:
- Delete branches
- Force update branches
- Rewrite branch history
- Force push
- Modify an existing branch without user approval

CRITICAL POLICY:
If a branch requested by the user already exists:
- Do not inspect it.
- Do not verify it.
- Do not reuse it.
- Do not modify it.
Immediately output:

HUMAN_INPUT_REQUIRED:
Branch '<name>' already exists.
How would you like to proceed?

OUTPUT FORMAT:

NEXT_COMMAND:
<single git command>

OR

HUMAN_INPUT_REQUIRED:
<question>
"""
)


def coordinator_node(state: GitState):

    # =====================================
    # Human Response -> Replan
    # =====================================

    human_response = state.get(
        "human_response",
        ""
    )

    if human_response:

        return {
            "replan": True,
            "waiting_for_user": False,
            "todo_complete": False,
            "current_todo": 0,
            "todos": [],
        }

    # =====================================
    # Normal Workflow
    # =====================================

    todos = state.get("todos", [])

    current_idx = state.get(
        "current_todo",
        0
    )

    # Advance todo if observer completed it

    if state.get(
        "todo_complete",
        False
    ):

        current_idx += 1

        if current_idx >= len(todos):

            return {
                "current_todo": current_idx,
                "done": True,
            }

    # Safety

    if current_idx >= len(todos):

        return {
            "done": True
        }

    current_todo = todos[current_idx]

    # =====================================
    # Coordinator Context
    # =====================================

    todo_context = SystemMessage(
        content=f"""
CURRENT TODO:
{current_todo}

TODO PROGRESS:
{current_idx + 1}/{len(todos)}

LAST COMMAND OUTPUT:
{state.get("last_output", "")}

OBSERVER REVIEW:
{state.get("review", "NONE")}
"""
    )

    messages = [
        COORDINATOR_PROMPT,
        todo_context,
    ] + state["messages"]

    response = llm.invoke(messages)

    content = response.content.strip()

    # =====================================
    # Human Escalation
    # =====================================

    if content.startswith(
        "HUMAN_INPUT_REQUIRED:"
    ):

        question = (
            content
            .replace(
                "HUMAN_INPUT_REQUIRED:",
                ""
            )
            .strip()
        )

        print(
            "\n===== HUMAN INPUT REQUIRED ====="
        )

        print(question)

        return {
            "waiting_for_user": True,
            "question": question,
            "current_todo": current_idx,
            "todo_complete": False,
        }

    # =====================================
    # Normal Command Generation
    # =====================================

    print("\n===== COORDINATOR =====")
    print(f"TODO: {current_todo}")
    print(content)

    return {
        "messages": [response],
        "current_todo": current_idx,
        "todo_complete": False,
    }