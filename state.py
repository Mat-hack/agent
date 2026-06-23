from typing import TypedDict
from langchain_core.messages import BaseMessage

class GitState(TypedDict):
    messages: list[BaseMessage]

    # Which repository this request operates on (multi-repo: one per project).
    repo_path: str

    todos: list[str]
    current_todo: int

    done: bool

    last_return_code: int
    last_output: str

    # Structured conflict signal from the toolbox (deterministic routing).
    conflict: bool
    conflict_files: list[str]

    review: str
    todo_complete: bool

    # Ordinal -> SHA resolved from the latest `git log` output, so the
    # coordinator never has to re-count "commit N" from raw history.
    commit_map: dict[int, str]

    waiting_for_user: bool
    question: str
    replan: bool
    original_request: str
    human_response: str
