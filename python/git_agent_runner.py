"""Bridge to the LangGraph git agent (root graph.py).

Branch / push / merge / cherry-pick requests are delegated to the existing git
agent — the planner -> coordinator -> executor -> observer -> human loop, with
its conflict handling — run against a specific project's repo. This module just
feeds it one request and drives its human-in-the-loop to completion, mirroring
main.py's outer loop.
"""
import sys
from pathlib import Path

# The git agent's modules (graph, planner, coordinator, ...) live in the repo
# root, one level above python/. Append (don't prepend) so python/'s own
# modules still win — only the root-unique modules resolve there.
_ROOT = Path(__file__).resolve().parent.parent


def _load_app():
    if str(_ROOT) not in sys.path:
        sys.path.append(str(_ROOT))
    import graph  # root graph.py — builds the compiled LangGraph `app`
    return graph.app


def _default_ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ''


def run_git_agent(request: str, repo_path: str, ask=None) -> dict:
    """Run the git agent for `request` against `repo_path` to completion."""
    from langchain_core.messages import HumanMessage

    app = _load_app()
    ask = ask or _default_ask

    state = {
        'messages': [HumanMessage(content=request)],
        'repo_path': repo_path,
        'done': False,
        'last_return_code': 0,
        'last_output': '',
        'conflict': False,
        'conflict_files': [],
        'original_request': request,
        'review': 'NONE',
        'replan': False,
        'human_response': '',
        'waiting_for_user': False,
        'todo_complete': False,
        'question': '',
    }

    while True:
        result = app.invoke(state)

        if result.get('done') and not result.get('waiting_for_user'):
            return result

        if result.get('waiting_for_user'):
            answer = ask('> ')
            state = {
                **result,
                'messages': result['messages'] + [HumanMessage(content=answer)],
                'human_response': answer,
                'waiting_for_user': False,
                'replan': False,
                'done': False,
            }
            continue

        return result
