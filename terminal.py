import subprocess

PROJECT_PATH = "/path/to/your/project"  # Change this


def execute_command(command: str) -> str:
    """
    Execute a shell command in the project directory and
    return stdout/stderr.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=PROJECT_PATH,
            capture_output=True,
            text=True,
        )

        return f"""
COMMAND:
{command}

RETURN CODE:
{result.returncode}

STDOUT:
{result.stdout}

STDERR:
{result.stderr}
"""
    except Exception as e:
        return f"ERROR: {e}"