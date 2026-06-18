from typing import TypedDict
from langchain_core.messages import BaseMessage

class GitState(TypedDict):
    messages: list[BaseMessage]

    todos: list[str]
    current_todo: int

    done: bool

    last_return_code: int

    review: str
    todo_complete: bool

    waiting_for_user: bool
    question: str
    replan: bool
    original_request: str
    human_response: str

   