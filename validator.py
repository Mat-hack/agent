"""
RAW git command mode + guardrail.

Self-contained: everything needed for `run_command(repo_path, cmd)` and
nothing else. The LLM emits a real git command string; `validate_command`
decides whether it may run, then `_run` executes it without shell=True.

Safety envelope (no delete / rename / rebase / reset / force):
  1. SUBCOMMAND must be on the allowlist (read set + safe write set).
  2. Every FLAG must be on that subcommand's allowlist.
  3. No shell metacharacters, no global flags before the subcommand,
     plus a hard-blocked token set as belt-and-suspenders.
"""

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime


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
# RAW COMMAND MODE — allowlist-first guardrail
# ---------------------------------------------------------------------------

CMD_READ_SUBS = {"log", "status", "diff", "show", "rev-parse", "cat-file", "ls-files"}
READ_BLOCKED_FLAGS = {"--output", "-o", "--ext-diff", "--textconv"}


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

CMD_VALUE_FLAGS = {"-m", "--message", "-n", "--max-count", "--mainline"}


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

    Allows a safe read+write envelope expressed as a literal command.
    Rejects everything else.
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

    Mutating commands still surface conflicts structurally so the
    coordinator's conflict routing works.
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