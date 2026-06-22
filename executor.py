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
    result = validator.run_command(repo_path, command)
    output = result.summary()

    print("\n===== OUTPUT =====")
    print(output)

    return {
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