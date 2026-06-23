"""Natural-language front door for the DB agent.

You type ONE request — which may describe many things at once (a project, a
package, several value types, services) — and this decomposes it into the
individual create/update operations, in dependency order, then makes the
respective API calls.

    python nl_agent.py
    > create project ekam with a package ecom, value types Customer (id, name,
      email) and Order (id, total), and a payments service

Or one-shot:
    python nl_agent.py "create a Customer value type in package ecom"
"""
import sys

from langchain_groq import ChatGroq

import config
import git_agent_runner
import git_orchestrator
from agent import DbOpAborted, DbOpFailed, run_db_op
from backend_client import BackendApiClient, BackendApiError
from decomposer import decompose


def available_entities() -> list[str]:
    """Entity names we can act on — one per schema in the cache."""
    return sorted(p.stem for p in config.SCHEMA_CACHE_DIR.glob('*.json'))


_ROUTE_SCHEMA = {
    'title': 'emit_route',
    'type': 'object',
    'properties': {
        # One request can ask for several PHASES; set every one it mentions.
        # They run in this order: build -> generate -> branch -> push.
        'build': {
            'type': 'boolean',
            'description': 'create/define records (a project, packages, value '
                           'types, services).',
        },
        'generate': {
            'type': 'boolean',
            'description': "materialise a project's code via generateProject "
                           'and set up its repo.',
        },
        'branch': {
            'type': 'boolean',
            'description': 'create a NEW git branch.',
        },
        'push': {
            'type': 'boolean',
            'description': 'push a branch to origin.',
        },
        'git_other': {
            'type': ['string', 'null'],
            'description': 'any OTHER git operation (merge, cherry-pick) as a '
                           'plain instruction for the git agent; null if none.',
        },
        'project_id': {
            'type': ['string', 'null'],
            'description': 'the id of the project (lowercase, as named). null '
                           'if none.',
        },
        'branch_name': {
            'type': ['string', 'null'],
            'description': 'the branch name if the user stated one, else null.',
        },
    },
    'required': ['build', 'generate', 'branch', 'push', 'git_other',
                 'project_id', 'branch_name'],
}

_ROUTER_SYSTEM = (
    'You break one request into the PHASES it asks for. A single request may '
    'ask for several — set a flag true for EACH phase it mentions:\n'
    '- build: it creates/defines records (a project, packages, value types, '
    'services).\n'
    '- generate: it asks to generate / materialise the project code ("generate '
    'it", "generate reader1").\n'
    '- branch: it asks to create a NEW git branch.\n'
    '- push: it asks to push a branch.\n'
    '- git_other: any OTHER git op (merge, cherry-pick) — put it as plain text, '
    'else null.\n'
    'Also extract project_id (lowercase, as named) and branch_name (only if a '
    'name is stated).\n'
    'Examples:\n'
    '- "generate ekam" -> generate=true, project_id=ekam.\n'
    '- "create project x with a value type y, then generate it and make a '
    'branch feat from developer and push it" -> build=true, generate=true, '
    'branch=true, push=true, project_id=x, branch_name=feat.\n'
    '- "give me a branch for qwert" -> branch=true, project_id=qwert.\n'
    'NOTE: "developer" and "main" are git branch names, never a project id. If '
    'no branch name is stated, leave branch_name null (never use the base '
    'branch as the name).'
)


def classify_action(nl: str) -> dict:
    llm = ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=0)
    structured = llm.with_structured_output(_ROUTE_SCHEMA, method='function_calling')
    return structured.invoke([('system', _ROUTER_SYSTEM), ('human', f'Request: {nl}')])


def confirm(question: str) -> bool:
    try:
        return input(f'{question} (y/n) ').strip().lower() == 'y'
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def run_plan(nl: str, client: BackendApiClient) -> list[dict]:
    """Decompose one request into ordered ops, show the plan, confirm once,
    then execute every step in order. Returns the backend results.

    Parent-first ordering plus the live re-grounding inside run_db_op means a
    record created by an earlier step is already visible to a later step that
    references it — no manual id threading needed.
    """
    entities = available_entities()
    steps = decompose(nl, entities)

    if not steps:
        print('Nothing to do — could not derive any operations from that.')
        return []

    print(f'\nPlan — {len(steps)} operation(s):')
    for i, step in enumerate(steps, 1):
        print(f'  {i}. {step["operation"]} {step["entity"]}: {step["intent"]}')

    if not confirm('\nExecute this plan?'):
        print('Aborted — nothing was sent.')
        return []

    results = []
    for i, step in enumerate(steps, 1):
        print(f'\n----- step {i}/{len(steps)}: {step["operation"]} {step["entity"]} -----')
        try:
            # Approve once for the whole plan, so each step runs unattended.
            result = run_db_op(
                step['operation'],
                step['entity'],
                step['intent'],
                client=client,
                auto_approve=True,
            )
            results.append(result)
        except (BackendApiError, DbOpFailed) as error:
            detail = error.data if isinstance(error, BackendApiError) else error
            print(f'\nStep {i} failed: {detail}')
            print(f'Stopped. {len(results)}/{len(steps)} step(s) completed.')
            return results

    print(f'\nDone — all {len(steps)} operation(s) completed.')
    return results


def _report_push(pushed: dict | None) -> None:
    """Print push status so success/failure is never mistaken for the commit."""
    if not pushed:
        return
    ok = [b for b, v in pushed.items() if v]
    bad = [b for b, v in pushed.items() if not v]
    if ok:
        print(f'Pushed to origin: {", ".join(ok)}.')
    if bad:
        print(f'NOT pushed (rejected/failed): {", ".join(bad)} — see the error '
              'above (e.g. GitHub push protection / credentials).')


def _print_git_outcome(outcome: dict) -> None:
    action = outcome.get('action')
    if action == 'init':
        if outcome.get('ok'):
            print(f'\nRepo initialised (main + developer) and generated code '
                  f'committed to developer at {outcome.get("repo_path")}.')
            _report_push(outcome.get('pushed'))
        else:
            print(f'\nRepo init/commit issue:\n{outcome.get("detail")}')
    elif action == 'update':
        if outcome.get('ok'):
            print(f'\nGenerated code committed to developer at '
                  f'{outcome.get("repo_path")}.')
        else:
            print(f'\nDeveloper already up to date (nothing new to commit).')
        _report_push(outcome.get('pushed'))
    else:
        print(f'\nNo git action taken ({outcome.get("reason", "skipped")}).')


def run_generate(project_id: str | None, client: BackendApiClient) -> None:
    """Call generateProject for a named project, then do its repo work:
    first time -> init repo (main + developer); thereafter -> commit to
    developer so it stays the up-to-date trunk."""
    project_id = project_id or _prompt('Which project to generate? ')
    if not project_id:
        print('No project specified.')
        return

    print(f'\nGenerating project "{project_id}"...')
    try:
        result = client.generate_project(project_id)
    except BackendApiError as error:
        print(f'generateProject failed: {error.data or error}')
        return
    print('Generated.' if not result else f'Generated: {result}')

    _print_git_outcome(git_orchestrator.finalize_generate(project_id))


def _project_repo(project_id: str | None) -> tuple[str | None, str | None]:
    """Resolve a project's repo, asking for the id if missing. Returns
    (project_id, repo_path) or (id, None) if there's no repo yet."""
    project_id = project_id or (_prompt('Which project? ').lower() or None)
    if not project_id:
        print('No project specified.')
        return None, None
    repo_path = git_orchestrator.repo_path_for(project_id)
    if not git_orchestrator.has_repo(repo_path):
        print(f'\nNo repo for "{project_id}" at {repo_path} — generate the '
              'project first.')
        return project_id, None
    return project_id, repo_path


def run_branch_push(project_id: str | None, branch_name: str | None,
                    do_branch: bool, do_push: bool) -> None:
    """Create a branch and/or push it, via the LangGraph git agent."""
    project_id, repo_path = _project_repo(project_id)
    if not repo_path:
        return

    if do_branch and not branch_name:
        branch_name = _prompt('Branch name? ')
    if not branch_name:
        branch_name = _prompt('Which branch to push? ') if do_push else ''
    if not branch_name:
        print('No branch name given.')
        return

    if do_branch and do_push:
        request = (f'create a branch named {branch_name} from developer and '
                   f'push branch {branch_name} to origin')
    elif do_branch:
        request = f'create a branch named {branch_name} from developer'
    else:
        request = f'push branch {branch_name} to origin'

    print(f'\n--- handing to git agent on {repo_path} ---')
    git_agent_runner.run_git_agent(request, repo_path)


def run_git_request(request: str, project_id: str | None) -> None:
    """Hand an arbitrary git instruction (merge, cherry-pick) to the git agent."""
    project_id, repo_path = _project_repo(project_id)
    if not repo_path:
        return
    print(f'\n--- handing to git agent on {repo_path} ---')
    git_agent_runner.run_git_agent(request, repo_path)


def _prompt(label: str) -> str:
    try:
        return input(label).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ''


def main() -> int:
    if not config.GROQ_API_KEY:
        print('GROQ_API_KEY env var is required.')
        return 1

    client = BackendApiClient()

    one_shot = ' '.join(sys.argv[1:]).strip()
    requests = [one_shot] if one_shot else None

    while True:
        if requests is not None:
            if not requests:
                return 0
            nl = requests.pop(0)
        else:
            try:
                nl = input('\n> ').strip()
            except (EOFError, KeyboardInterrupt):
                print()  # newline so the shell prompt isn't glued on
                return 0
            if nl.lower() in {'exit', 'quit', 'q'}:
                return 0
        if not nl:
            continue

        try:
            route = classify_action(nl)
            pid = (route.get('project_id') or '').strip().lower() or None

            # Run whatever phases the request asked for, in dependency order.
            did_something = False
            if route.get('build'):
                run_plan(nl, client)
                did_something = True
            if route.get('generate'):
                run_generate(pid, client)
                did_something = True
            if route.get('git_other'):
                run_git_request(route['git_other'], pid)
                did_something = True
            elif route.get('branch') or route.get('push'):
                run_branch_push(pid, route.get('branch_name'),
                                route.get('branch', False), route.get('push', False))
                did_something = True

            if not did_something:
                # Nothing classified — treat it as a build request by default.
                run_plan(nl, client)
        except DbOpAborted:
            print('Aborted — nothing was sent.')
        except Exception as error:  # routing / decomposition / network — keep the REPL alive
            text = str(error)
            if 'did not call a tool' in text or 'tool_use_failed' in text:
                # The model answered with prose instead of a tool call — the
                # input was too vague or a fragment. Ask for a clearer request.
                print("I couldn't understand that — please rephrase it as one "
                      'clear request (and paste it on a single line).')
            else:
                print(f'Error: {error}')


if __name__ == '__main__':
    sys.exit(main())
