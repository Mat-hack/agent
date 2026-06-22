import re

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from state import GitState

from dotenv import load_dotenv
load_dotenv()

# Planning must be deterministic — the same request should yield the same
# plan every time, so temperature is pinned to 0.
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


# ---------------------------------------------------------------------------
# Repository conventions the planner may assume without asking or investigating.
# ---------------------------------------------------------------------------

REPO_RULES = SystemMessage(
    content="""
REPOSITORY CONVENTIONS (assume these; never plan a todo to discover them):

- `developer` is the integration branch. `main` is the clean base.
- A branch source that is not stated defaults to `developer`
  (e.g. "create branch X" means "create branch X from developer").
- "commit N" means the Nth commit on `developer`, unless another branch
  is named in the request.
- "the branch I'm at" / "current branch" / "where am I" refers to the
  currently checked-out branch. It is readable directly — never ask the
  user which branch they mean.
"""
)


# ---------------------------------------------------------------------------
# Planner specification.
# ---------------------------------------------------------------------------

PLANNER_PROMPT = SystemMessage(
    content="""
ROLE

You are a Git task planner. You convert one natural-language request into an
ordered list of atomic todos for a downstream executor. You ONLY plan:
- You never run anything.
- You never output git commands.
- You never ask the user a question.
- You never explain your reasoning.

EXECUTION MODEL

The executor downstream can perform exactly these operations. Plan only in
terms of these — a todo must map to exactly ONE of them:

READS (inspection):
- show the current branch
- list branches
- show the log / commits of a branch (this reveals every commit hash at once)
- show status, or a diff
- show the files contained in a commit

WRITES (mutation):
- create a branch from a ref/commit
- commit staged changes to `developer`
- cherry-pick one or more commits onto the current branch
- take specific files from a commit or branch (partial pick)
- merge one branch into another
- push a named branch to the remote (origin) — never force

CONFLICT HANDLING (only when a conflict is already in progress):
- resolve a conflict (keep ours / keep theirs)
- abort a cherry-pick or merge

UNSUPPORTED operations: delete a branch, rename a branch, rebase, reset,
stash, tag, revert, force-push. For a request that needs an unsupported
operation, still emit a SINGLE todo that plainly names it (e.g. "Delete
branch X"). The executor will decline it. NEVER invent multi-step
workarounds to simulate an unsupported operation.

PLANNING PRINCIPLES

1. Produce the SHORTEST correct plan. One todo = one git operation.
2. Add a read/discovery todo ONLY when an identifier required for a later
   step (such as a commit hash) is not given and cannot be assumed from the
   repository conventions. If the request already contains everything
   needed, plan execution todos only.
3. Combine lookups that one command answers. Multiple commits on the SAME
   branch are found with one log read — use a single todo for them, never
   one todo per commit.
4. Order todos by dependency: discover before use, create before operate on.
5. Never plan a todo whose action is "ask", "clarify", or "confirm with the
   user". Clarification is handled by the executor, not the plan.

OUTPUT CONTRACT

If the request requires any repository action, output ONLY:

TODOS:
1. <todo>
2. <todo>

If the request is purely conceptual and needs no repository action at all
(e.g. "what is git rebase"), output ONLY:

DONE

Output nothing else — no prose, no markdown, no commands, no commentary.

EXAMPLES

User: make branch rett from developer commit 3
TODOS:
1. Find commit 3 hash on developer
2. Create branch rett at that hash

User: check branch
TODOS:
1. Show the current branch

User: show last 3 commits
TODOS:
1. Show the last 3 commits on developer

User: make branch demarioio from developer with commit 2 and commit 4
TODOS:
1. Find commit 2 and commit 4 hashes on developer
2. Create branch demarioio at commit 2 hash
3. Cherry-pick commit 4 onto demarioio

User: make a branch qwemty from commit 5 and push it
TODOS:
1. Find commit 5 hash on developer
2. Create branch qwemty at commit 5 hash
3. Push branch qwemty to remote

User: push potato
TODOS:
1. Push branch potato to remote

User: force push potato
TODOS:
1. Force push branch potato to remote

User: delete branch alice
TODOS:
1. Delete branch alice

User: what is git rebase
DONE
"""
)


def _build_messages(state: GitState):
    """Assemble the message list for either a fresh plan or a replan."""
    base = [REPO_RULES, PLANNER_PROMPT]

    if not state.get("replan"):
        return base + state["messages"]

    # Replan: the user corrected or changed the request mid-workflow. Carry
    # the full prior conversation so already-gathered facts (hashes, branch
    # names, file lists) are reused rather than re-investigated, and any
    # todos already completed are not repeated.
    continuation = HumanMessage(
        content=f"""
The workflow was interrupted and a new plan is required.

ORIGINAL REQUEST:
{state.get("original_request", "")}

USER CORRECTION / UPDATE:
{state.get("human_response", "")}

The conversation above already contains command outputs (commit hashes,
branch names, file lists). Reuse those facts directly — do NOT plan todos
that re-investigate information already visible above, and do NOT repeat
steps that already succeeded. Plan only the remaining work needed to
satisfy the corrected request.
"""
    )
    return base + state["messages"] + [continuation]


def planner_node(state: GitState):

    messages = _build_messages(state)
    response = llm.invoke(messages)

    content = response.content.strip()

    print("\n===== PLANNER =====")
    print(content)

    todos = extract_todos(content)

    if todos:
        print("\n===== TODOS =====")
        print(todos)
        return {
            "messages": [response],
            "todos": todos,
            "current_todo": 0,
            "done": False,
            "replan": False,
            "human_response": "",
        }

    # No todos parsed. Either an explicit DONE (conceptual question, nothing
    # to do) or genuinely unusable output.
    if re.search(r"\bDONE\b", content.upper()):
        return {
            "messages": [response],
            "todos": [],
            "current_todo": 0,
            "done": True,
            "replan": False,
            "human_response": "",
        }

    raise ValueError(f"Planner produced no todos and no DONE:\n{content}")


def extract_todos(text: str):
    """Pull numbered todo lines ('1.', '2)', etc.) into a clean list."""
    todos = []
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"^\d+[.)]\s+(.*)", line)
        if match:
            todo = match.group(1).strip()
            if todo:
                todos.append(todo)
    return todos
