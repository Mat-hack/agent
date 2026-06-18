import subprocess
import re

from langchain_core.messages import HumanMessage

from state import GitState

from dotenv import load_dotenv
load_dotenv()
REPO_PATH = "/Users/mukulsharma/testing/Sentimental_analysis_main"


def extract_command(text: str) -> str | None:
    """
    Extract command from:

    NEXT_COMMAND:
    git log developer --oneline
    """

    match = re.search(
        r"NEXT_COMMAND:\s*(.+)",
        text,
        re.DOTALL
    )

    if not match:
        return None

    return match.group(1).strip()


def executor_node(state: GitState):

    latest_message = state["messages"][-1]

    command = extract_command(latest_message.content)

    if not command:
        return {
            "messages": [
                HumanMessage(
                    content="ERROR: No NEXT_COMMAND found."
                )
            ],
            "last_return_code": 1
        }

    print(f"\n>>> RUNNING: {command}\n")
    print("\n===== EXECUTING =====")
    print(command)

    result = subprocess.run(
        command,
        cwd=REPO_PATH,
        shell=True,
        capture_output=True,
        text=True
    )

    output = result.stdout.strip()

    if not output:
        output = result.stderr.strip()

    print("\n===== OUTPUT =====")
    print(output)

    return {
        "messages": [
            HumanMessage(
                content=f"""
COMMAND:
{command}

OUTPUT:
{output}
"""
            )
        ],
        "last_return_code": result.returncode
    }
