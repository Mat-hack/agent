import re

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

from state import GitState

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")


# The coordinator writes the literal git command itself; the executor's
# guardrail (valiator.run_command) validates it before running.
COORDINATOR_PROMPT = SystemMessage(
    content="""
You are a Git Workflow Coordinator operating on a real repository.

You complete ONE todo at a time by writing exactly ONE real git command, or
escalating to a human. Output ONLY one of:

1. A git command to run:
   COMMAND: git <command>

2. An escalation to a human. NEVER escalate blind — first gather evidence by
   running a read command (git diff / git show / git status / git log), THEN
   on the next turn escalate with this exact 3-part shape so the human gets a
   grounded picture and a clear steer, not a bare menu:
   HUMAN_INPUT_REQUIRED:
   FINDINGS: <what the repo actually shows, quoting the read you just ran —
             e.g. what `git diff`/`git show --stat` revealed and WHY you are
             stuck. Be concrete; do not speculate.>
   RECOMMENDATION: <the ONE option you would pick and a one-line reason.>
   OPTIONS:
   1. <option + the exact git command it maps to>
   2. <option + command>
   3. <option + command>   (include only the options that genuinely apply)

3. Give up on the current todo cleanly:
   ABORT: <one-line reason>
   Use this when the todo cannot be done and there is no path forward — a
   forbidden operation the human will not redirect, or a question you have
   already asked once and the human declined. This ends the workflow.
   NEVER ask the same unanswerable question more than once.

For a cherry-pick whose commit is already contained in the target branch
(an ancestor), DO NOT skip or abort on your own judgement — just emit the
normal `git cherry-pick <hash>` command. The executor detects the no-op
ancestor case deterministically and reports it as done; never try to
pre-judge it yourself.

ALLOWED COMMANDS (this is the full safe envelope — anything else is rejected
by the guardrail before it runs, so do not attempt it):
- Reads: git log, git status, git diff, git show, git rev-parse, git branch
  (listing only, e.g. `git branch --list`), git ls-files, git cat-file
- Branch: git branch <name> <start-point>   (create only)
- Checkout: git checkout -b <name> <start>, git checkout <branch>,
  git checkout <ref> -- <files>, git checkout --ours|--theirs -- <files>
- Stage/commit: git add ..., git commit -m "<msg>"
- Cherry-pick: git cherry-pick <hash>...,
  git cherry-pick --continue|--abort|--skip
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
- A RESOLVED COMMITS block (ordinal -> SHA) may be provided below. When the
  todo references "commit N", use the SHA mapped to N there verbatim. If the
  block is empty and you need a SHA, run `git log <branch> --oneline` first.
- Never repeat a command already run for the SAME todo with the same result.
- `developer` is the integration branch; `main` is the clean base.
- BRANCH YOU ACT ON: cherry-pick and commit always apply to the CURRENTLY
  CHECKED-OUT branch. `git branch B <start>` creates B but does NOT switch to
  it — HEAD stays where it was. So to cherry-pick/commit ONTO B you MUST run
  `git checkout B` first. When a plan creates a branch and then works on it,
  create it with `git checkout -b B <start>` so you create AND switch in one
  step, and never pick onto the wrong branch.

CONFLICT HANDLING (a command reported [CONFLICT]):
- Run `git diff <file>` on the first conflicted file ONCE to gather evidence,
  then on the next turn escalate using the FINDINGS/RECOMMENDATION/OPTIONS
  shape above, where the options are:
    1. Keep ours   -> git checkout --ours -- <files>
    2. Keep theirs -> git checkout --theirs -- <files>
    3. Abort       -> git cherry-pick --abort  (or git merge --abort)
- After they choose ours/theirs, resolve with that checkout, then
  git add <files>, then git cherry-pick --continue (or git merge --continue).

EMPTY CHERRY-PICK (a command stopped with "nothing to commit, working tree
clean" while a cherry-pick is in progress):
- This is NOT a conflict. It means the commit's changes are ALREADY present
  on this branch, so the patch is empty. `git cherry-pick --continue` will
  keep failing — do NOT repeat it.
- First run `git show <commit> --stat` ONCE to see what the commit was meant
  to change (evidence), then escalate using the FINDINGS/RECOMMENDATION/
  OPTIONS shape, with:
    FINDINGS: the patch is empty because the branch already contains these
              changes (quote the git show output).
    RECOMMENDATION: skip it — the branch is already in the desired state.
    OPTIONS:
    1. Skip the empty commit -> git cherry-pick --skip
    2. Abort the cherry-pick -> git cherry-pick --abort
- After they choose, emit `git cherry-pick --skip` or `git cherry-pick
  --abort` accordingly.

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

    commit_map = state.get("commit_map") or {}
    if commit_map:
        commits_note = "\n".join(
            f"{n} -> {sha}" for n, sha in sorted(commit_map.items())
        )
    else:
        commits_note = "NONE (run `git log <branch> --oneline` to populate)"

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

RESOLVED COMMITS (ordinal -> SHA, from the latest git log):
{commits_note}

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
