import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from langsmith import traceable
from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """Standardized result of a code execution attempt."""

    exit_code: int
    stdout: str
    stderr: str
    passed: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def generate_thread_id(prefix: Optional[str] = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{timestamp}" if prefix else timestamp


def is_timeout_result(result: ExecutionResult) -> bool:
    text = f"{result.error or ''} {result.stderr or ''}".lower()
    return "timeout" in text or "timed out" in text


@traceable(name="execution.sandbox.run_code")
def run_code_in_sandbox(
    *,
    sandbox: Any,
    code: str,
    execution_timeout_seconds: int,
) -> ExecutionResult:
    execution = sandbox.run_code(
        code,
        timeout=execution_timeout_seconds,
        request_timeout=execution_timeout_seconds,
    )
    execution_data = json.loads(execution.to_json())
    logs = execution_data.get("logs") or {}
    error = execution_data.get("error") or {}
    error_name = error.get("name") if isinstance(error, dict) else None
    error_value = error.get("value") if isinstance(error, dict) else ""
    timeout = "timeout" in f"{error_name or ''} {error_value or ''}".lower()

    return ExecutionResult(
        exit_code=0 if error_name is None else -1,
        stdout="".join(logs.get("stdout") or []),
        stderr="".join(logs.get("stderr") or []) or str(error_value or ""),
        passed=error_name is None,
        error="Timeout" if timeout else error_name,
        metadata={"execution": execution_data},
    )


@traceable(name="execution.local.run_isolated_code")
def run_isolated_code(code: str, task_id: str, timeout_seconds: int = 30) -> ExecutionResult:
    if not code.strip():
        return ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr="Code generation failed: empty code payload.",
            passed=False,
            error="EmptyCodeError",
            metadata={"method": "local", "task_id": task_id},
        )

    solutions_dir = Path("./data/solutions")
    solutions_dir.mkdir(parents=True, exist_ok=True)

    # Task-specific filename ensures no collisions in parallel runs
    file_path = solutions_dir / f"{task_id}.py"

    # Flash (overwrite) or create the solution file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)

    metadata = {
        "method": "local",
        "task_id": task_id,
        "solution_path": str(file_path),
    }

    try:
        result = subprocess.run(
            [sys.executable, str(file_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=os.environ.copy(),
        )
        return ExecutionResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            passed=result.returncode == 0,
            metadata=metadata,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr=f"Execution timed out ({timeout_seconds}s limit).",
            passed=False,
            error="Timeout",
            metadata=metadata,
        )
    except Exception as e:
        return ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr=str(e),
            passed=False,
            error=type(e).__name__,
            metadata=metadata,
        )


__all__ = [
    "ExecutionResult",
    "generate_thread_id",
    "is_timeout_result",
    "run_code_in_sandbox",
    "run_isolated_code",
]
