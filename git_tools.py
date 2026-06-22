"""
Git toolbox + guard.

This is the ONLY place that mutates a repository.

The LLM never emits a raw mutating command. It picks a tool name + typed
args (see TOOLS at the bottom); the actual `git` invocation lives here and
runs without shell=True. Read-only inspection (log/status/diff/...) is the
one exception and goes through run_inspect(), which is allowlist-gated.
"""

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime

# Primary integration branch. "main" is the clean base and is never committed
# to directly by the agent.
DEVELOPER_BRANCH = "developer"
MAIN_BRANCH = "main"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GitResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    conflict: bool = False
    files: list = field(default_factory=list)

    def summary(self) -> str:
        """Human + LLM readable one-block summary."""
        out = self.stdout or self.stderr or "(no output)"
        head = "OK" if self.ok else ("CONFLICT" if self.conflict else "FAILED")
        extra = ""
        if self.conflict and self.files:
            extra = "\nCONFLICTED FILES:\n" + "\n".join(self.files)
        return f"[{head}]\n{out}{extra}"


# ---------------------------------------------------------------------------
# Guard: what is allowed / blocked
# ---------------------------------------------------------------------------

# Free-form inspection is allowed only for these read-only subcommands.
READ_ALLOWLIST = {
    "log",
    "status",
    "diff",
    "show",
    "rev-parse",
    "branch",        # constrained below to listing only
    "cat-file",
    "ls-files",
}

# Never allowed, anywhere, from any path.
# NOTE: this only governs the free-form INSPECT path. Push is also exposed
# as a real tool (push_branch, below) — but only as a plain, non-force push;
# git itself refuses non-fast-forward updates unless --force is passed, and
# --force/-f never appear in any arg list this codebase builds.
BLOCKED_TOKENS = {
    "-D",            # branch -D
    "-f",            # branch -f / push -f (force)
    "--force",
    "--hard",        # reset --hard
    "-fd",           # clean -fd
    "push",          # push is not a read operation; use the push_branch tool
}

# branch is read-only only for these flag shapes.
BRANCH_READ_FLAGS = {"--list", "-a", "-r", "-v", "-vv", "--show-current", "--contains"}


# Argument values that must look like a ref/name/hash, never a flag.
import re as _re

_NAME_RE = _re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+-]*$")
_REF_RE = _re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+^~@{}-]*$")

# Which string args of each tool are refs/names/paths (must be validated as
# such) vs free text (e.g. a commit message, which may contain anything).
_REF_ARGS = {"name", "base_ref", "ref", "source", "target", "branch"}
_LIST_REF_ARGS = {"hashes"}
_PATH_ARGS = {"files"}
_FREE_ARGS = {"message", "remote", "side", "kind"}  # validated elsewhere / arbitrary


def _bad_ref(v) -> str | None:
    """Return an error string if v is not a safe ref/name, else None."""
    if not isinstance(v, str) or not v:
        return "must be a non-empty string"
    if v.startswith("-"):
        return f"'{v}' looks like a flag (leading '-') — refused"
    if not _REF_RE.match(v):
        return f"'{v}' is not a valid git ref/name"
    if ".." in v:
        return "'..' is not allowed in a ref/name"
    return None


def _bad_path(v) -> str | None:
    """Return an error string if v is not a safe repo-relative path, else None."""
    if not isinstance(v, str) or not v:
        return "must be a non-empty string"
    if v.startswith("-"):
        return f"'{v}' looks like a flag (leading '-') — refused"
    if v.startswith("/") or v.startswith("~"):
        return "absolute paths are not allowed"
    if ".." in v.split("/"):
        return "path traversal ('..') is not allowed"
    return None


def validate_tool_args(name: str, args: dict) -> str | None:
    """
    Reject flag-injection and path-traversal before any git runs.

    This is the guarantee that no tool can be coerced into delete/force/reset
    behaviour by smuggling a flag through an argument value
    (e.g. create_branch(name="-D")).
    """
    if not isinstance(args, dict):
        return "args must be an object"

    for key, value in args.items():
        if key in _LIST_REF_ARGS:
            if not isinstance(value, list) or not value:
                return f"{key} must be a non-empty list"
            for item in value:
                err = _bad_ref(item)
                if err:
                    return f"{key}: {err}"
        elif key in _PATH_ARGS:
            if not isinstance(value, list) or not value:
                return f"{key} must be a non-empty list"
            for item in value:
                err = _bad_path(item)
                if err:
                    return f"{key}: {err}"
        elif key in _REF_ARGS:
            err = _bad_ref(value)
            if err:
                return f"{key}: {err}"
        elif key in _FREE_ARGS:
            continue
        else:
            return f"unexpected argument '{key}' for {name}"

    return None


def is_safe_inspect(cmd: str) -> tuple[bool, str]:
    """Validate a free-form inspection command. Returns (ok, reason)."""
    if any(ch in cmd for ch in [";", "&", "|", "`", "$", ">", "<", "\n"]):
        return False, "shell metacharacters are not allowed"

    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        return False, f"unparseable command: {e}"

    if not parts or parts[0] != "git":
        return False, "only 'git' commands are allowed"

    if len(parts) < 2:
        return False, "no git subcommand"

    sub = parts[1]
    if sub not in READ_ALLOWLIST:
        return False, f"'{sub}' is not a read-only subcommand"

    rest = set(parts[2:])
    if rest & BLOCKED_TOKENS:
        return False, "command contains a blocked token"

    # `branch` may only be used to list, never to create/delete/move.
    if sub == "branch":
        non_flags = [p for p in parts[2:] if not p.startswith("-")]
        bad_flags = [p for p in parts[2:] if p.startswith("-") and p not in BRANCH_READ_FLAGS]
        if non_flags or bad_flags:
            return False, "branch may only be used for listing (use the create_branch tool to create)"

    return True, ""


# ---------------------------------------------------------------------------
# Low-level runner (no shell)
# ---------------------------------------------------------------------------

def _run(repo_path: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )


def _conflicted_files(repo_path: str) -> list[str]:
    """Files currently in an unmerged/conflicted state."""
    cp = _run(repo_path, ["status", "--porcelain"])
    files = []
    for line in cp.stdout.splitlines():
        if len(line) < 3:
            continue
        code = line[:2]
        # Any 'U', or AA / DD, indicates an unmerged path.
        if "U" in code or code in {"AA", "DD"}:
            files.append(line[3:].strip())
    return files


def log_action(repo_path: str, message: str) -> None:
    """
    Append a natural-language line to the per-repo agent log.

    Stored under .git/ so it is never tracked, staged, or able to cause
    spurious merge/cherry-pick conflicts.
    """
    try:
        git_dir = os.path.join(repo_path, ".git")
        base = git_dir if os.path.isdir(git_dir) else repo_path
        path = os.path.join(base, "agent_log")
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a") as f:
            f.write(f"[{stamp}] {message}\n")
    except OSError:
        pass  # logging must never break the workflow


# ---------------------------------------------------------------------------
# Tools (mutating). Each returns a GitResult.
# ---------------------------------------------------------------------------

def init_repo(repo_path: str, remote: str | None = None) -> GitResult:
    """New project: init, create clean `main`, branch off `developer`."""
    os.makedirs(repo_path, exist_ok=True)

    if os.path.isdir(os.path.join(repo_path, ".git")):
        return GitResult(False, stderr=f"{repo_path} is already a git repo")

    _run(repo_path, ["init"])
    _run(repo_path, ["checkout", "-b", MAIN_BRANCH])
    # An empty initial commit so branches have a base.
    _run(repo_path, ["commit", "--allow-empty", "-m", "chore: initial commit"])
    cp = _run(repo_path, ["checkout", "-b", DEVELOPER_BRANCH])

    if remote:
        _run(repo_path, ["remote", "add", "origin", remote])

    ok = cp.returncode == 0
    if ok:
        log_action(repo_path, f"Initialized repo with {MAIN_BRANCH} + {DEVELOPER_BRANCH} branches")
    return GitResult(ok, cp.stdout, cp.stderr)


def commit_to_developing(repo_path: str, message: str) -> GitResult:
    """Stage everything and commit on the developer branch."""
    _run(repo_path, ["checkout", DEVELOPER_BRANCH])
    _run(repo_path, ["add", "-A"])
    cp = _run(repo_path, ["commit", "-m", message])
    ok = cp.returncode == 0
    if ok:
        log_action(repo_path, f"Committed to {DEVELOPER_BRANCH}: {message}")
    return GitResult(ok, cp.stdout, cp.stderr)


def create_branch(repo_path: str, name: str, base_ref: str = DEVELOPER_BRANCH) -> GitResult:
    """Create branch `name` from `base_ref`. Refuses to clobber existing."""
    exists = _run(repo_path, ["rev-parse", "--verify", "--quiet", f"refs/heads/{name}"])
    if exists.returncode == 0:
        return GitResult(False, stderr=f"branch '{name}' already exists")

    cp = _run(repo_path, ["branch", name, base_ref])
    ok = cp.returncode == 0
    if ok:
        log_action(repo_path, f"Created branch '{name}' from {base_ref}")
    return GitResult(ok, cp.stdout or f"created {name}", cp.stderr)


def cherry_pick(repo_path: str, hashes: list[str]) -> GitResult:
    """Cherry-pick one or more commits onto the current branch."""
    cp = _run(repo_path, ["cherry-pick", *hashes])
    if cp.returncode == 0:
        log_action(repo_path, f"Cherry-picked {', '.join(hashes)}")
        return GitResult(True, cp.stdout, cp.stderr)

    conflicts = _conflicted_files(repo_path)
    if conflicts:
        return GitResult(False, cp.stdout, cp.stderr, conflict=True, files=conflicts)
    return GitResult(False, cp.stdout, cp.stderr)


def checkout_files(repo_path: str, ref: str, files: list[str]) -> GitResult:
    """Bring specific files from `ref` into the working tree (partial pick)."""
    cp = _run(repo_path, ["checkout", ref, "--", *files])
    ok = cp.returncode == 0
    if ok:
        log_action(repo_path, f"Checked out {len(files)} file(s) from {ref}: {', '.join(files)}")
    return GitResult(ok, cp.stdout or f"checked out {len(files)} file(s)", cp.stderr)


def merge(repo_path: str, source: str, target: str) -> GitResult:
    """Merge `source` into `target`. Reports conflict instead of leaving a mess."""
    _run(repo_path, ["checkout", target])
    cp = _run(repo_path, ["merge", "--no-ff", source])
    if cp.returncode == 0:
        log_action(repo_path, f"Merged {source} into {target}")
        return GitResult(True, cp.stdout, cp.stderr)

    conflicts = _conflicted_files(repo_path)
    if conflicts:
        return GitResult(False, cp.stdout, cp.stderr, conflict=True, files=conflicts)
    return GitResult(False, cp.stdout, cp.stderr)


def resolve_conflict(repo_path: str, side: str) -> GitResult:
    """
    Resolve all conflicted files by taking one side, then continue.

    side: 'ours' (current branch) or 'theirs' (incoming).
    """
    if side not in {"ours", "theirs"}:
        return GitResult(False, stderr="side must be 'ours' or 'theirs'")

    files = _conflicted_files(repo_path)
    if not files:
        return GitResult(False, stderr="no conflict in progress")

    _run(repo_path, ["checkout", f"--{side}", "--", *files])
    _run(repo_path, ["add", *files])

    # Continue whichever operation is in progress.
    git_dir = os.path.join(repo_path, ".git")
    if os.path.exists(os.path.join(git_dir, "CHERRY_PICK_HEAD")):
        cp = _run(repo_path, ["cherry-pick", "--continue"])
    elif os.path.exists(os.path.join(git_dir, "MERGE_HEAD")):
        cp = _run(repo_path, ["commit", "--no-edit"])
    else:
        cp = _run(repo_path, ["status"])

    ok = cp.returncode == 0
    if ok:
        log_action(repo_path, f"Resolved conflict keeping '{side}' for {len(files)} file(s)")
    return GitResult(ok, cp.stdout, cp.stderr)


def push_branch(repo_path: str, branch: str) -> GitResult:
    """
    Push a local branch to 'origin'. Never force.

    The remote is always 'origin' (not an argument — eliminates remote-name
    injection entirely). git itself refuses a non-fast-forward update unless
    --force is passed, and --force/-f never appear in this arg list, so this
    cannot silently overwrite remote history; a rejected push just fails.
    """
    cp = _run(repo_path, ["push", "origin", branch])
    ok = cp.returncode == 0
    if ok:
        log_action(repo_path, f"Pushed '{branch}' to origin")
    return GitResult(ok, cp.stdout, cp.stderr)


def abort(repo_path: str, kind: str) -> GitResult:
    """Abort an in-progress cherry-pick or merge."""
    if kind not in {"cherry-pick", "merge"}:
        return GitResult(False, stderr="kind must be 'cherry-pick' or 'merge'")
    cp = _run(repo_path, [kind, "--abort"])
    ok = cp.returncode == 0
    if ok:
        log_action(repo_path, f"Aborted {kind}")
    return GitResult(ok, cp.stdout or f"aborted {kind}", cp.stderr)


def run_inspect(repo_path: str, cmd: str) -> GitResult:
    """Run an allowlisted read-only command. Never mutates."""
    ok, reason = is_safe_inspect(cmd)
    if not ok:
        return GitResult(False, stderr=f"blocked: {reason}")

    parts = shlex.split(cmd)
    cp = _run(repo_path, parts[1:])
    return GitResult(cp.returncode == 0, cp.stdout, cp.stderr)


# ---------------------------------------------------------------------------
# RAW COMMAND MODE
#
# An alternative to the toolbox: the LLM emits a real git command string and
# this guardrail decides whether it may run. Same safety envelope as the
# tools (no delete / rename / rebase / reset / force), but the command is the
# model's literal output — useful as a transparent, demoable artifact.
#
# Defence is allowlist-first, on two axes:
#   1. SUBCOMMAND must be on the allowlist (read set + safe write set).
#   2. Every FLAG must be on that subcommand's allowlist — an unknown or
#      dangerous flag is rejected, never merely "not blocked".
# Plus: no shell metacharacters, no global flags before the subcommand
# (kills `-c config=...` injection and `-C /other/dir` redirects), and a
# hard-blocked token set as belt-and-suspenders.
# ---------------------------------------------------------------------------

# Read subcommands: cannot mutate state. Flags are allowed freely except a
# small set that can write files or execute external programs.
CMD_READ_SUBS = {"log", "status", "diff", "show", "rev-parse", "cat-file", "ls-files"}
READ_BLOCKED_FLAGS = {"--output", "-o", "--ext-diff", "--textconv"}

# Write subcommands: each maps to an explicit allowlist of permitted flags.
# Any flag not listed here is rejected. Destructive flags (-D, --force,
# --amend, -m/-M rename, ...) are simply absent, so they cannot pass.
CMD_WRITE_FLAGS = {
    "branch":      {"--list", "-a", "-r", "-v", "-vv", "--show-current",
                    "--contains", "--merged", "--no-merged"},
    "checkout":    {"-b", "--ours", "--theirs", "-t", "--track"},
    "cherry-pick": {"--continue", "--abort", "--skip", "-n", "--no-commit",
                    "-x", "-m", "--mainline"},
    "commit":      {"-m", "--message", "-a", "--all", "--no-edit", "--allow-empty"},
    "merge":       {"--no-ff", "--ff-only", "--no-edit", "--abort", "--continue",
                    "-m", "--message"},
    "push":        {"-u", "--set-upstream"},
    "add":         {"-A", "--all", "-u", "--update"},
}

# Flags whose FOLLOWING token is a literal value (e.g. a commit message that
# may itself start with '-'), so that token is not flag-checked.
CMD_VALUE_FLAGS = {"-m", "--message", "-n", "--max-count", "--mainline"}

# Belt-and-suspenders: never allowed in any write command, whatever a
# per-subcommand allowlist might be widened to in future. Note '-m' is
# deliberately absent (it is a commit message flag); branch rename via
# 'branch -m' is still blocked because '-m' is not in branch's allowlist.
CMD_HARD_BLOCKED = {
    "-f", "--force", "--force-with-lease", "--hard", "--amend", "--mirror",
    "-D", "-d", "--delete", "-M", "--move", "--orphan", "--detach", "-B",
    "-c", "-C", "--copy",
}


def _split_flag_token(tok: str) -> tuple[list[str], bool]:
    """
    Normalise one flag token into its constituent flag names + whether its
    value is attached (so the next token is NOT a separate value).

      --message=foo  -> (['--message'], True)
      --no-edit      -> (['--no-edit'], False)
      -am            -> (['-a', '-m'], False)   # message is the next token
      -amwip         -> (['-a', '-m'], True)    # message attached as 'wip'
      -B             -> (['-B'], False)
    """
    if tok.startswith("--"):
        name = tok.split("=", 1)[0]
        return [name], ("=" in tok)

    # Short-flag cluster: expand char by char until a value-taking flag.
    chars = tok[1:]
    flags = []
    attached = False
    for i, ch in enumerate(chars):
        f = "-" + ch
        flags.append(f)
        if f in CMD_VALUE_FLAGS:
            attached = i < len(chars) - 1  # remainder of token is its value
            break
    return flags, attached


def validate_command(cmd: str) -> tuple[bool, str]:
    """
    Validate a full git command for RAW mode. Returns (ok, reason).

    Allows the same safe read+write envelope as the toolbox, expressed as a
    literal command. Rejects everything else.
    """
    if any(ch in cmd for ch in [";", "&", "|", "`", "$", ">", "<", "\n"]):
        return False, "shell metacharacters are not allowed"

    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        return False, f"unparseable command: {e}"

    if not parts or parts[0] != "git":
        return False, "only 'git' commands are allowed"
    if len(parts) < 2:
        return False, "no git subcommand"

    # No global flags between `git` and the subcommand: blocks `-c key=val`
    # config injection and `-C /dir` / `--git-dir` repo redirection.
    if parts[1].startswith("-"):
        return False, "global flags before the subcommand are not allowed"

    sub = parts[1]
    rest = parts[2:]

    if sub in CMD_READ_SUBS:
        for tok in rest:
            if not tok.startswith("-"):
                continue
            name = tok.split("=", 1)[0]
            if name in READ_BLOCKED_FLAGS:
                return False, f"flag '{name}' is not allowed for {sub}"
        return True, ""

    if sub not in CMD_WRITE_FLAGS:
        return False, f"'{sub}' is not an allowed subcommand"

    allowed = CMD_WRITE_FLAGS[sub]
    after_ddash = False
    skip_next = False

    for tok in rest:
        if skip_next:
            skip_next = False
            continue
        if tok == "--":
            after_ddash = True
            continue
        if after_ddash:
            continue  # pathspecs, not flags
        if not tok.startswith("-"):
            continue  # positional: ref / path / value

        names, attached_value = _split_flag_token(tok)
        for name in names:
            if name in CMD_HARD_BLOCKED:
                return False, f"flag '{name}' is blocked"
            if name not in allowed:
                return False, f"flag '{name}' is not allowed for {sub}"
        if not attached_value and any(n in CMD_VALUE_FLAGS for n in names):
            skip_next = True

    # push must be a plain 'git push origin <branch>' — never a refspec
    # delete (':branch') or force-via-'+'.
    if sub == "push":
        positional = [t for t in rest if not t.startswith("-")]
        for t in positional:
            if ":" in t or t.startswith("+"):
                return False, "refspec push (':' or '+') is not allowed"

    return True, ""


def run_command(repo_path: str, cmd: str) -> GitResult:
    """
    RAW mode: validate an LLM-written git command, then run it (no shell).

    Mutating commands still surface conflicts structurally, exactly like the
    cherry_pick / merge tools, so the coordinator's conflict routing works
    in either mode.
    """
    ok, reason = validate_command(cmd)
    if not ok:
        return GitResult(False, stderr=f"blocked: {reason}")

    parts = shlex.split(cmd)
    sub = parts[1]
    cp = _run(repo_path, parts[1:])

    if cp.returncode == 0:
        if sub not in CMD_READ_SUBS:
            log_action(repo_path, f"Ran: {cmd}")
        return GitResult(True, cp.stdout, cp.stderr)

    # Non-zero: detect an in-progress conflict for pick/merge.
    if sub in {"cherry-pick", "merge"}:
        conflicts = _conflicted_files(repo_path)
        if conflicts:
            return GitResult(False, cp.stdout, cp.stderr, conflict=True, files=conflicts)

    return GitResult(False, cp.stdout, cp.stderr)


# ---------------------------------------------------------------------------
# Tool registry — the surface the coordinator chooses from.
# ---------------------------------------------------------------------------

TOOLS = {
    "init_repo": {
        "fn": init_repo,
        "args": ["repo_path", "remote?"],
        "desc": "New project: init repo + create main and developer branches.",
    },
    "commit_to_developing": {
        "fn": commit_to_developing,
        "args": ["repo_path", "message"],
        "desc": "Stage all and commit on the developer branch.",
    },
    "create_branch": {
        "fn": create_branch,
        "args": ["repo_path", "name", "base_ref?"],
        "desc": "Create a branch from base_ref (default: developer). Refuses to overwrite.",
    },
    "cherry_pick": {
        "fn": cherry_pick,
        "args": ["repo_path", "hashes[]"],
        "desc": "Cherry-pick one or more commit hashes onto the current branch.",
    },
    "checkout_files": {
        "fn": checkout_files,
        "args": ["repo_path", "ref", "files[]"],
        "desc": "Bring specific files from a commit/branch into the working tree.",
    },
    "merge": {
        "fn": merge,
        "args": ["repo_path", "source", "target"],
        "desc": "Merge source into target; reports conflict instead of leaving a mess.",
    },
    "push_branch": {
        "fn": push_branch,
        "args": ["repo_path", "branch"],
        "desc": "Push a local branch to origin. Never force — non-fast-forward updates are rejected by git itself.",
    },
    "resolve_conflict": {
        "fn": resolve_conflict,
        "args": ["repo_path", "side"],
        "desc": "Resolve all conflicts taking 'ours' or 'theirs', then continue.",
    },
    "abort": {
        "fn": abort,
        "args": ["repo_path", "kind"],
        "desc": "Abort an in-progress 'cherry-pick' or 'merge'.",
    },
}


def tool_catalog() -> str:
    """Render the toolbox for the coordinator prompt."""
    lines = []
    for name, meta in TOOLS.items():
        args = ", ".join(a for a in meta["args"] if a != "repo_path")
        lines.append(f"- {name}({args}): {meta['desc']}")
    return "\n".join(lines)


def dispatch_tool(name: str, repo_path: str, args: dict) -> GitResult:
    """Validate and call a tool by name with JSON args (repo_path injected)."""
    if name not in TOOLS:
        return GitResult(False, stderr=f"unknown tool: {name}")

    err = validate_tool_args(name, args)
    if err:
        return GitResult(False, stderr=f"rejected: {err}")

    fn = TOOLS[name]["fn"]
    try:
        return fn(repo_path=repo_path, **args)
    except TypeError as e:
        return GitResult(False, stderr=f"bad args for {name}: {e}")
