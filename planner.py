from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage

from state import GitState

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)
REPO_RULES = SystemMessage(
    content="""
Repository Rules:

- developer is the primary branch.
- If a user says "create branch X",
  assume "create branch X from developer".
- If a user says "commit 4",
  assume commit 4 on developer.
- Do not create todos to determine the source branch
  when developer can be assumed.
- Never create investigation todos for assumptions
  already covered by repository rules.
"""
)

PLANNER_PROMPT = SystemMessage(
    content="""
You are a Git Task Planner.

Your only job is to break a git request into atomic todos.

RULES:

- Output ONLY todos.
- Do NOT output commands.
- Do NOT output NEXT_COMMAND.
- Do NOT output explanations.
- Do NOT output reasoning.
- Do NOT output markdown.
- Every todo must require exactly one git command.

OUTPUT FORMAT:

TODOS:
1. <todo>
2. <todo>
3. <todo>

EXAMPLE

User:
make branch rett from developer commit 3

Response:

TODOS:
1. Find commit 3 hash on developer
2. Create branch rett at that hash

User:
what is git rebase

Response:

DONE
"""
)


def planner_node(state: GitState):

    if state.get("replan"):

        messages = [
            REPO_RULES,
            PLANNER_PROMPT,
            HumanMessage(
    content=f"""
The workflow was interrupted and replanning is required.

Original Request:
{state["original_request"]}

User Correction / Update:
{state["human_response"]}

Generate a new todo list using the updated request.
"""
)
        ]

    else:

        messages = [
            REPO_RULES,
            PLANNER_PROMPT,
        ] + state["messages"]

    response = llm.invoke(messages)

    print("\n===== PLANNER =====")
    print(response.content)

    content = response.content.strip()

    if content == "DONE":
        return {
            "messages": [response],
            "todos": [],
            "current_todo": 0,
            "done": True,
        }

    todos = extract_todos(content)

    print("\n===== TODOS =====")
    print(todos)

    if not todos:
        raise ValueError(
            f"Planner produced invalid output:\n{content}"
        )

    return {
    "messages": [response],
    "todos": todos,
    "current_todo": 0,
    "done": False,
    "replan": False,
    "human_response": "",
}

import re


def extract_todos(text: str):

    todos = []

    for line in text.splitlines():

        line = line.strip()

        if re.match(r"^\d+\.", line):

            todo = re.sub(
                r"^\d+\.\s*",
                "",
                line
            )

            todos.append(todo)

    return todos