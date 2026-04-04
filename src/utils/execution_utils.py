import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

class ExecutionResult(BaseModel):
    """Standardized result of a code execution attempt."""
    exit_code: int
    stdout: str
    stderr: str
    passed: bool
    error: Optional[str] = None


def generate_thread_id(prefix: Optional[str] = None) -> str:
    right_now = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{right_now}" if prefix else right_now

def run_isolated_code(code: str, task_id: str) -> ExecutionResult:
    """
    Writes code to a task-specific file in data/solutions and executes it.
    Persists the file for debugging.
    """
    if not str(code or "").strip():
        return ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr="Code generation failed: empty code payload.",
            passed=False,
            error="EmptyCodeError",
        )

    solutions_dir = Path("./data/solutions")
    solutions_dir.mkdir(parents=True, exist_ok=True)
    
    # Task-specific filename ensures no collisions in parallel runs
    file_path = solutions_dir / f"{task_id}.py"
    
    # Flash (overwrite) or create the solution file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    
    try:
        # Run in a subprocess for environment isolation
        result = subprocess.run(
            [sys.executable, str(file_path)],
            capture_output=True,
            text=True,
            timeout=30,
            env=os.environ.copy()  # Propagate Notion/Google keys
        )
        return ExecutionResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            passed=result.returncode == 0
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr="Execution timed out (30s limit).",
            passed=False,
            error="Timeout"
        )
    except Exception as e:
        return ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr=str(e),
            passed=False,
            error=type(e).__name__
        )


__all__ = ["ExecutionResult", "generate_thread_id", "run_isolated_code"]
