"""
Executor.

The coordinator hands us a single literal git command:

    COMMAND: git <command>

The command is validated by the guardrail (validator.run_command) before it
runs. There is no shell=True path.
"""

import os
import re

from langchain_core.messages import HumanMessage

from state import GitState
import validator

from dotenv import load_dotenv
load_dotenv()

# Fallback when a request did not carry an explicit repo_path.
DEFAULT_REPO_PATH = os.getenv(
    "REPO_PATH",
    "/Users/mukulsharma/testing/Sentimental_analysis_main",
)


def _extract_command(text: str) -> str | None:
    """Pull the git command out of a `COMMAND: git ...` directive."""
    m = re.search(r"COMMAND:\s*(.+)", text, re.DOTALL)
    if not m:
        return None
    # The command is a single line; don't swallow any trailing prose.
    return m.group(1).splitlines()[0].strip()


# `commit <40-hex>` (full log) or a leading `<7-40 hex>` token (--oneline).
_FULL_SHA_RE = re.compile(r"^commit\s+([0-9a-f]{40})\b", re.MULTILINE)
_ONELINE_SHA_RE = re.compile(r"^([0-9a-f]{7,40})\b", re.MULTILINE)


def _parse_commit_map(stdout: str) -> dict[int, str]:
    """
    Build an ordinal -> SHA map from `git log` output, in the order shown.
    "commit N" means the Nth line, matching how the user counts them.
    """
    shas = _FULL_SHA_RE.findall(stdout)
    if not shas:
        shas = _ONELINE_SHA_RE.findall(stdout)
    return {i + 1: sha for i, sha in enumerate(shas)}


def executor_node(state: GitState):

    repo_path = state.get("repo_path") or DEFAULT_REPO_PATH

    latest = state["messages"][-1].content
    command = _extract_command(latest)

    if not command:
        return {
            "messages": [HumanMessage(content="ERROR: No COMMAND directive found.")],
            "last_return_code": 1,
            "last_output": "no command",
        }

    print(f"\n>>> COMMAND: {command}\n")

    # Deterministic no-op guard: cherry-picking a commit that is already an
    # ancestor of HEAD re-applies changes that are already present (an empty
    # or duplicate commit). Detect it here with `merge-base --is-ancestor`
    # rather than trusting the model to reason about ancestry. Only the plain
    # single-commit form (`git cherry-pick <hash>`) — never --continue/--abort
    # or a range/multi-pick, which we let run normally.
    parts = command.split()
    if (
        len(parts) == 3
        and parts[1] == "cherry-pick"
        and not parts[2].startswith("-")
        and ".." not in parts[2]
        and validator.is_ancestor(repo_path, parts[2], "HEAD")
    ):
        output = (
            f"[OK]\nCommit {parts[2]} is already an ancestor of HEAD; its "
            "changes are already present. Cherry-pick skipped as a no-op."
        )
        print("\n===== OUTPUT =====")
        print(output)
        return {
            "messages": [HumanMessage(content=f"COMMAND:\n{command}\n\nRESULT:\n{output}")],
            "last_return_code": 0,
            "last_output": output,
            "conflict": False,
            "conflict_files": [],
        }

    result = validator.run_command(repo_path, command)
    output = result.summary()

    print("\n===== OUTPUT =====")
    print(output)

    update = {
        "messages": [
            HumanMessage(
                content=f"""
COMMAND:
{command}

RESULT:
{output}
"""
            )
        ],
        "last_return_code": 0 if result.ok else 1,
        "last_output": output,
        "conflict": result.conflict,
        "conflict_files": result.files,
    }

    # When we just listed history, persist the ordinal -> SHA map so the
    # coordinator references exact hashes instead of re-counting "commit N".
    if result.ok and command.split()[1:2] == ["log"]:
        commit_map = _parse_commit_map(result.stdout)
        if commit_map:
            update["commit_map"] = commit_map

    return update