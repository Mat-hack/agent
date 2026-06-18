from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import subprocess

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

REPO_PATH = "/Users/mukulsharma/testing/Sentimental_analysis_main"

@tool
def run_git_command(command: str) -> str:
    """Runs a git command on the repo and returns the output."""

    print(f"\n RUNNING: {command}\n")

    result = subprocess.run(
        command,
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
        shell=True
    )

    output = result.stdout or result.stderr

    print(output)     

    return output

checkpointer = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[run_git_command],
    checkpointer=checkpointer,
    prompt=SystemMessage(content="""You are a Git assistant operating on a real repository.

Rules:

* Use the run_git_command tool whenever repository information is needed.
* Never invent repository state, branch names, commits, files, or changes.
* Verify information through tool calls before answering.
* Use the minimum number of tool calls required.
* Do not perform exploratory investigation unless explicitly requested.
* Do not execute destructive operations unless explicitly requested.
* If a command fails, explain the error and stop unless the next action is obvious.
* Ask for clarification when the user's request is ambiguous.
* Keep responses concise and focused on the task.
* Do not explain your reasoning process.
* just give the result don't tell me output let the tool work
"""
    )
)

config = {"configurable": {"thread_id": "git-session-1"}}

while True:
    user_input = input("You: ")
    if not user_input:
        continue

    response = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config
    )

    # print(response["messages"][-1].content)