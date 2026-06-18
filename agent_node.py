from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import subprocess

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

REPO_PATH = "/Users/mukulsharma/testing/Sentimental_analysis_main"


class GitState(TypedDict):
    messages: list
    done: bool


SYSTEM_PROMPT = SystemMessage(
    content="""
You are a git assistant operating on a real repository.
You work exactly like a human developer would in a terminal.

HOW YOU WORK:
- You execute one command at a time
- You read the output before deciding the next command
- You follow the natural git workflow a developer would follow
- For example, before committing: check status → check branch → switch if needed → add → commit → push
- Checking branch is just the first step, always continue with the full workflow
- For a commit task the full flow is: check branch → git status → git add . → git commit -m "message" → git push
- Only output DONE after the final push is confirmed successful

STRICT RULES:
1. Output ONLY a single raw shell command per response
2. No markdown, no explanation, no code fences, no extra text
3. Only do what the user explicitly asked — nothing more
4. Do not push unless asked
5. Do not create branches unless asked
6. Do not fix unrelated issues you notice along the way
7. If a command fails, output exactly: DONE
8. When the task is fully complete, output exactly: DONE

WORKFLOW RULES:
- Always check current branch before switching or committing
- Always check status before adding files
- If already on the right branch, skip the switch
- Read output carefully — if something already exists or is already done, skip it
TASK EXAMPLES:
- "branch from commit 3 on developer" means:
  1. git log --oneline developer
  2. identify the 3rd commit hash from the top
  3. git checkout -b <new-branch-name> <that-hash>

- Checking status and branch is only needed for commit/push tasks, not for branch creation tasks.
- Read the user request carefully and only run relevant commands for that task.

OUTPUT FORMAT:
- One command only
- Raw text, no symbols before or after
- When done: DONE
"""
)


def git_node(state: GitState):
    messages = [SYSTEM_PROMPT] + state["messages"]

    response = llm.invoke(messages)

    command = response.content.strip()

    command = response.content.strip()

# handle DONE mixed in with command
    if "DONE" in command:
        lines = [l for l in command.split("\n") if l.strip() != "DONE"]
        command = "\n".join(lines).strip()
        if not command:
            print("\n>>> TASK COMPLETE\n")
            return {"messages": [response], "done": True}

    print(f"\n>>> RUNNING: {command}\n")

    result = subprocess.run(
        command,
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
        shell=True
    )

    output = result.stdout or result.stderr

    if result.returncode != 0:
        print(f">>> ERROR: {output}")
        return {
            "messages": [response, HumanMessage(content=f"Command failed:\n{output}\nStop and report the error.")],
            "done": True  # stop the loop on failure
        }

    print(output)

    return {
        "messages": [
            response,
            HumanMessage(
                content=f"Command Output:\n{output}"
            )
        ],
        "done": False
    }


def route(state: GitState):
    if state.get("done"):
        return "end"
    return "continue"


graph = StateGraph(GitState)

graph.add_node("git", git_node)

graph.set_entry_point("git")

graph.add_conditional_edges(
    "git",
    route,
    {
        "continue": "git",
        "end": END
    }
)

checkpointer = MemorySaver()

app = graph.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": "git-session-1"
    }
}


while True:
    user_input = input("You: ")

    app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "done": False
        },
        config
    )