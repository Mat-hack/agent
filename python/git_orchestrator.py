"""Git side of the orchestration.

After a project is generated, this decides what to do with its repository:

  - FIRST TIME (no repo yet): ask the human to confirm, optionally take a
    remote, then init the repo (main + developer + initial commit) and commit
    the freshly generated code onto developer.

  - ALREADY A REPO: the project already exists in git, so just cut a branch
    (from developer by default) for someone to work on.

The actual git work is delegated to the vetted, no-shell tools in
api_call/git_tools.py — this module only orchestrates them and talks to the
human; it never builds a git command itself.
"""
import os
import sys
from pathlib import Path

import config

# git_tools lives in the sibling api_call/ package, outside python/.
_API_CALL_DIR = Path(__file__).resolve().parent.parent / 'api_call'
if str(_API_CALL_DIR) not in sys.path:
    sys.path.insert(0, str(_API_CALL_DIR))

import git_tools  # noqa: E402


def repo_path_for(project_id: str) -> str:
    """Local path where this project's repo lives (or will be created)."""
    return os.path.join(str(config.PROJECTS_ROOT), project_id)


def has_repo(repo_path: str) -> bool:
    return os.path.isdir(os.path.join(repo_path, '.git'))


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ''


def finalize_generate(project_id: str, *, ask=_ask) -> dict:
    """Repo work that follows a generateProject call.

    First time (no repo): init main + developer and commit the generated code
    to developer. Existing repo: commit the freshly regenerated code to
    developer so it stays the always-updated trunk. Branches are NOT cut here —
    those are separate git queries handled by the git agent.
    """
    repo_path = repo_path_for(project_id)

    if not has_repo(repo_path):
        return _first_time_init(project_id, repo_path, ask)

    _ensure_gitignore(repo_path)
    _untrack_ignored(repo_path)
    res = git_tools.commit_to_developing(
        repo_path, f'chore: regenerate project {project_id}'
    )

    # Always push (even if the commit was a no-op): the local branches may have
    # commits that were never pushed. If there's no origin yet, offer to set one.
    if not _has_origin(repo_path):
        remote = ask('No origin set. Remote URL to push to (blank to skip): ') or None
        if remote:
            git_tools._run(repo_path, ['remote', 'add', 'origin', remote])

    pushed = _push_branches(repo_path) if _has_origin(repo_path) else None

    return {
        'action': 'update',
        'ok': res.ok,
        'repo_path': repo_path,
        'pushed': pushed,
        'detail': res.summary(),
    }


def _has_origin(repo_path: str) -> bool:
    cp = git_tools._run(repo_path, ['remote'])
    return 'origin' in cp.stdout.split()


def _push_branches(repo_path: str, branches=None) -> dict:
    """Push the given branches to origin; return {branch: ok}. Reports each
    failure (auth / no remote) without raising."""
    branches = branches or (git_tools.MAIN_BRANCH, git_tools.DEVELOPER_BRANCH)
    results = {b: git_tools.push_branch(repo_path, b) for b in branches}
    for branch, r in results.items():
        if not r.ok:
            print(f'  push {branch} failed: {r.stderr.strip()}')
    return {b: r.ok for b, r in results.items()}


# Secret/env patterns we must guarantee are ignored so GitHub push protection
# can't reject the push. The `**/` prefix matches at the root AND every nested
# folder, so a single root .gitignore covers `apps/backend/worker/.env.template`
# and friends regardless of any per-folder .gitignore the generator ships.
# `.env.example` is kept since it's meant to be shared.
_SECRET_IGNORE_RULES = [
    '**/.env',
    '**/.env.*',
    '!**/.env.example',
    '*.pem',
    '*.key',
]


def _ensure_gitignore(repo_path: str) -> None:
    """Make sure the repo's .gitignore excludes secret/env files.

    Generated projects ship their OWN .gitignore that usually doesn't exclude
    .env.template, so we must APPEND any missing rules rather than skip when a
    file already exists. Called before any `git add -A`.
    """
    path = os.path.join(repo_path, '.gitignore')
    existing = ''
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = f.read()
        except OSError:
            existing = ''

    have = {line.strip() for line in existing.splitlines()}
    missing = [rule for rule in _SECRET_IGNORE_RULES if rule not in have]
    if not missing:
        return

    try:
        with open(path, 'a') as f:
            if existing and not existing.endswith('\n'):
                f.write('\n')
            f.write('\n# secrets / environment (added by orchestrator)\n')
            f.write('\n'.join(missing) + '\n')
    except OSError:
        pass  # a missing .gitignore must not break the flow


def _untrack_ignored(repo_path: str) -> None:
    """Drop any already-tracked files that are now ignored from the index, so
    the next commit doesn't re-include them (e.g. a .env.template that was
    committed before the ignore rule existed)."""
    cp = git_tools._run(repo_path, ['ls-files', '-ci', '--exclude-standard'])
    files = [line for line in cp.stdout.splitlines() if line.strip()]
    if files:
        git_tools._run(repo_path, ['rm', '--cached', '--', *files])


def _first_time_init(project_id: str, repo_path: str, ask) -> dict:
    print(f'\nNo git repo yet for project "{project_id}".')
    print(f'Expected location: {repo_path}')
    if not os.path.isdir(repo_path):
        print('(That directory does not exist — is PROJECTS_ROOT pointing at '
              'where generateProject writes its output?)')

    if ask('Initialise a new repo here and commit the generated code? (y/n) ').lower() != 'y':
        return {'action': 'none', 'reason': 'declined first-time init'}

    remote = ask('Remote URL to attach as origin (blank for none): ') or None

    init_res = git_tools.init_repo(repo_path, remote=remote)
    if not init_res.ok:
        return {'action': 'init', 'ok': False, 'detail': init_res.summary()}

    # Write .gitignore BEFORE the first commit so secret/env files (e.g.
    # .env.template with AWS keys) are never staged, and the push isn't
    # rejected by GitHub push protection. The generated project ships its own
    # .gitignore, so we append our rules; _untrack_ignored drops anything
    # already staged that the new rules now ignore.
    _ensure_gitignore(repo_path)
    _untrack_ignored(repo_path)

    commit_res = git_tools.commit_to_developing(
        repo_path, f'feat: generate project {project_id}'
    )

    # If a remote was given, push both branches so the GitHub repo actually
    # reflects the project — init_repo only wires up `origin`, it never pushes.
    pushed = _push_branches(repo_path) if remote else None

    return {
        'action': 'init',
        'ok': commit_res.ok,
        'repo_path': repo_path,
        'remote': remote,
        'pushed': pushed,
        'detail': commit_res.summary(),
    }
