import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
load_dotenv()
from graph import app

DEFAULT_REPO_PATH = os.getenv(
    "REPO_PATH",
    "/Users/mukulsharma/testing/Sentimental_analysis_main",
)

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in {"exit", "quit", "q"}:
        break

    repo_path = DEFAULT_REPO_PATH
    if user_input.lower().startswith("repo:"):
        first, _, rest = user_input[len("repo:"):].strip().partition(" ")
        repo_path, user_input = first, rest.strip()
        if not user_input:
            print("Provide a request after the repo path.")
            continue

    state = {
        "messages": [HumanMessage(content=user_input)],
        "repo_path": repo_path,
        "done": False,
        "last_return_code": 0,
        "last_output": "",
        "conflict": False,
        "conflict_files": [],
        "original_request": user_input,
        "review": "NONE",
        "replan": False,
        "human_response": "",
        "waiting_for_user": False,
        "todo_complete": False,
        "question": "",
    }

    while True:
        result = app.invoke(state)

        # task done normally
        if result.get("done") and not result.get("waiting_for_user"):
            print("\n=== TASK FINISHED ===\n")
            break

        # human input needed
        if result.get("waiting_for_user"):
            answer = input("> ").strip()

            state = {
                **result,
                "messages": result["messages"] + [HumanMessage(content=answer)],
                "human_response": answer,
                "waiting_for_user": False,
                "replan": False,
                "done": False,
            }
            continue

        break