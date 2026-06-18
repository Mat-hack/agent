from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
load_dotenv()
from graph import app

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in {"exit", "quit", "q"}:
        break

    state = {
        "messages": [HumanMessage(content=user_input)],
        "done": False,
        "last_return_code": 0,
        "original_request": user_input,
        "observer_result": "NONE",
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