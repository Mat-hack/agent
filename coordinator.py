import re

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

from state import GitState

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")


# The coordinator writes the literal git command itself; the executor's
# guardrail (git_tools.run_command) validates it before running.
COORDINATOR_PROMPT = SystemMessage(
    content="""
You are a Git Workflow Coordinator operating on a real repository.

You complete ONE todo at a time by writing exactly ONE real git command, or
escalating to a human. Output ONLY one of:

1. A git command to run:
   COMMAND: git <command>

2. An escalation to a human:
   HUMAN_INPUT_REQUIRED:
   <question>

3. Give up on the current todo cleanly:
   ABORT: <one-line reason>
   Use this when the todo cannot be done and there is no path forward — a
   forbidden operation the human will not redirect, or a question you have
   already asked once and the human declined. This ends the workflow.
   NEVER ask the same unanswerable question more than once.

ALLOWED COMMANDS (this is the full safe envelope — anything else is rejected
by the guardrail before it runs, so do not attempt it):
- Reads: git log, git status, git diff, git show, git rev-parse, git branch
  (listing only, e.g. `git branch --list`), git ls-files, git cat-file
- Branch: git branch <name> <start-point>   (create only)
- Checkout: git checkout -b <name> <start>, git checkout <branch>,
  git checkout <ref> -- <files>, git checkout --ours|--theirs -- <files>
- Stage/commit: git add ..., git commit -m "<msg>"
- Cherry-pick: git cherry-pick <hash>..., git cherry-pick --continue|--abort
- Merge: git merge --no-ff <branch>, git merge --continue|--abort
- Push: git push origin <branch>   (never --force)

FORBIDDEN (never emit — the guardrail will reject and it wastes a turn):
- Deleting or renaming branches (branch -d/-D/-m/-M)
- git reset, git rebase, git clean, git stash, git tag, git revert
- --force, --hard, --amend, force-push, refspec deletes (origin :branch)
- Any shell operators (; | & `` $()) or global flags (-c, -C, --git-dir)
For a todo that needs a forbidden operation, output HUMAN_INPUT_REQUIRED on
the FIRST turn explaining it is not supported — do NOT try a workaround. If
the human then declines or offers no alternative, output ABORT.

HARD RULES:
- Read the previous RESULT before acting. If a commit hash or branch name
  already appears in the conversation, use that exact value — NEVER guess a
  hash, and NEVER pass a bare ordinal like "4" as a hash. "commit N" means
  the Nth commit: find its real SHA in a prior log output and use that.
- Never repeat a command already run for the SAME todo with the same result.
- `developer` is the integration branch; `main` is the clean base.

CONFLICT HANDLING (a command reported [CONFLICT]):
- Run `git diff <file>` on the first conflicted file ONCE, then on the next
  turn output HUMAN_INPUT_REQUIRED asking the user to choose:
  1. Keep ours  2. Keep theirs  3. Abort
- After they choose, resolve with:
    git checkout --ours -- <files>   (then git add + git cherry-pick --continue)
  or git checkout --theirs -- <files> (then git add + git cherry-pick --continue)
  or git cherry-pick --abort / git merge --abort

NEVER ask the human "what would you like to do" for a todo that simply looks
something up — read it. Only escalate for a conflict choice, an already-
existing branch / destructive action, genuine ambiguity, or a forbidden op.

OUTPUT EXACTLY ONE DIRECTIVE. No explanations, no markdown.
"""
)


# Robust directive extraction: the model sometimes prefixes stray text
# (e.g. ": HUMAN_INPUT_REQUIRED:" or echoes the todo), so search rather than
# require the keyword at the very start.
_HUMAN_RE = re.compile(r"HUMAN_INPUT_REQUIRED:\s*(.*)", re.DOTALL)
_ABORT_RE = re.compile(r"ABORT:\s*(.*)", re.DOTALL)

# Pure-decline answers that mean "give up on this", as opposed to "no, do X
# instead". Kept to exact short phrases so corrections aren't misread.
_DECLINE = {
    "no", "n", "stop", "cancel", "abort", "quit", "skip", "skip it",
    "don't", "dont", "do not", "don't proceed", "dont proceed",
    "do not proceed", "leave it", "forget it", "nevermind", "never mind",
    "no thanks", "no thank you",
}


def _is_decline(text: str) -> bool:
    return text.strip().strip(".!").lower() in _DECLINE


def coordinator_node(state: GitState):

    # ----- Human response -> resume the SAME todo, do NOT discard the plan -----
    # The human answered a question about the current todo (conflict side,
    # a new branch name, a yes/no). The rest of the plan (remaining todos,
    # current_todo, conflict state) must survive — only a genuinely new/
    # changed request should ever go back through the planner, and nothing
    # here currently triggers that automatically.
    human_response = state.get("human_response", "")

    # Backstop: if the human's answer is a plain refusal ("no", "don't
    # proceed", "cancel"), they are abandoning the current todo — end the
    # workflow instead of resuming and re-asking the same question forever.
    if human_response and _is_decline(human_response):
        print("\n===== WORKFLOW ABANDONED =====")
        print(f"User declined; stopping. ({human_response!r})")
        return {
            "done": True,
            "waiting_for_user": False,
            "human_response": "",
            "last_output": f"Abandoned by user: {human_response}",
        }

    todos = state.get("todos", [])
    current_idx = state.get("current_todo", 0)

    # Advance when the observer marked the current todo complete.
    if state.get("todo_complete", False):
        current_idx += 1
        if current_idx >= len(todos):
            return {"current_todo": current_idx, "done": True}

    if current_idx >= len(todos):
        return {"done": True}

    current_todo = todos[current_idx]

    conflict_note = "NONE"
    if state.get("conflict"):
        files = ", ".join(state.get("conflict_files", []))
        conflict_note = f"ACTIVE — conflicted files: {files}"

    human_note = "NONE"
    if human_response:
        human_note = (
            f'The human just answered your last question: "{human_response}". '
            "Apply that answer directly to the CURRENT TODO now — e.g. use it "
            "as the branch name, conflict side, or confirmation. Do NOT ask "
            "the same question again."
        )

    todo_context = SystemMessage(
        content=f"""
CURRENT TODO:
{current_todo}

TODO PROGRESS:
{current_idx + 1}/{len(todos)}

LAST RESULT:
{state.get("last_output", "")}

CONFLICT STATE:
{conflict_note}

OBSERVER REVIEW:
{state.get("review", "NONE")}

HUMAN ANSWER:
{human_note}
"""
    )

    messages = [COORDINATOR_PROMPT, todo_context] + state["messages"]
    response = llm.invoke(messages)
    content = response.content.strip()

    # ----- Abort: the todo cannot be completed and there is no path forward -----
    abort_match = _ABORT_RE.search(content)
    if abort_match:
        reason = abort_match.group(1).strip() or "operation not supported"
        print("\n===== WORKFLOW ABANDONED =====")
        print(reason)
        return {
            "done": True,
            "waiting_for_user": False,
            "human_response": "",
            "last_output": f"Abandoned: {reason}",
        }

    # ----- Human escalation (robust to stray leading text) -----
    human_match = _HUMAN_RE.search(content)
    if human_match:
        question = human_match.group(1).strip()
        print("\n===== HUMAN INPUT REQUIRED =====")
        print(question)
        return {
            "waiting_for_user": True,
            "question": question,
            "current_todo": current_idx,
            "todo_complete": False,
            # Escalating again means this answer didn't resolve it (or a new
            # question is being asked) — executor/observer won't run this
            # turn, so clear it here; nothing downstream will consume it.
            "human_response": "",
        }

    print("\n===== COORDINATOR =====")
    print(f"TODO: {current_todo}")
    print(content)

    return {
        "messages": [response],
        "current_todo": current_idx,
        "todo_complete": False,
        # NOT cleared here: the observer still needs human_response to judge
        # the result against the human's corrected intent, not the stale
        # literal todo wording. The observer clears it once it's done.
    }
