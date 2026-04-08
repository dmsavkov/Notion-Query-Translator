import json
import os
import subprocess
import sys
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from e2b_code_interpreter import ALL_TRAFFIC, Sandbox, TimeoutException
from langsmith import traceable
from pydantic import BaseModel, ConfigDict, Field

try:
    from e2b_code_interpreter import SandboxNetworkOpts
except ImportError:  # pragma: no cover - fallback for older/newer SDK layout
    from e2b.sandbox.sandbox_api import SandboxNetworkOpts


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


@traceable(name="execution.sandbox.get_or_create")
def get_or_create_sandbox(
    sandbox_id: Optional[str],
    thread_id: str,
    task_id: str,
    template: str,
    timeout_seconds: int,
) -> Any:
    """Gets an existing sandbox or creates a new one, returning the Sandbox instance."""
    if sandbox_id:
        try:
            return Sandbox.connect(sandbox_id)
        except Exception:
            # Fallback to create if not found or connection failed
            pass

    return Sandbox.create(
        metadata={"thread_id": thread_id, "task_id": task_id},
        envs={key: value for key, value in os.environ.items() if key.startswith("NOTION_") and str(value).strip()},
        allow_internet_access=True,
        template=template,
        timeout=timeout_seconds,
        network=SandboxNetworkOpts(
            deny_out=[ALL_TRAFFIC],
            allow_out=["api.notion.com"],
            allow_public_traffic=False,
        ),
        api_key=os.getenv("E2B_API_KEY"),
    )


def kill_sandbox(sandbox_id: str) -> None:
    """Kills a sandbox by its ID if it exists."""
    if not sandbox_id:
        return
    try:
        sandbox = Sandbox.connect(sandbox_id)
        sandbox.kill()
    except Exception:
        pass


def _parse_json_if_string(value: Any) -> Any:
    if isinstance(value, str):
        with suppress(Exception):
            return json.loads(value)
    return value


def _join_log_lines(lines: Any) -> str:
    if isinstance(lines, str):
        return lines
    if isinstance(lines, list):
        return "".join(str(item) for item in lines)
    return ""


@traceable(name="execution.sandbox.run_code")
def run_code_in_sandbox(
    *,
    sandbox: Any,
    code: str,
    execution_timeout_seconds: int,
) -> ExecutionResult:
    try:
        execution = sandbox.run_code(
            code,
            timeout=execution_timeout_seconds,
            request_timeout=execution_timeout_seconds,
        )
    except TimeoutException as exc:
        return ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr=str(exc),
            passed=False,
            error="Timeout",
            metadata={"execution": None},
        )

    execution_data: Dict[str, Any] = {}
    with suppress(Exception):
        execution_data = json.loads(execution.to_json())

    logs_data = _parse_json_if_string(execution_data.get("logs") or {})
    if not isinstance(logs_data, dict):
        logs_data = {}

    error_data = _parse_json_if_string(execution_data.get("error"))
    if error_data is None:
        error_data = {}
    if not isinstance(error_data, dict):
        error_data = {"value": str(error_data)}

    # Fall back to object attributes when JSON payload shape differs across SDK versions.
    logs_obj = getattr(execution, "logs", None)
    stdout = _join_log_lines(logs_data.get("stdout"))
    stderr = _join_log_lines(logs_data.get("stderr"))
    if not stdout:
        stdout = _join_log_lines(getattr(logs_obj, "stdout", None))
    if not stderr:
        stderr = _join_log_lines(getattr(logs_obj, "stderr", None))

    error_name = error_data.get("name") if isinstance(error_data, dict) else None
    error_value = str(error_data.get("value") or "") if isinstance(error_data, dict) else ""

    error_obj = getattr(execution, "error", None)
    if error_obj is not None:
        error_name = error_name or getattr(error_obj, "name", None)
        if not error_value:
            error_value = str(getattr(error_obj, "value", "") or "")

    has_error = bool(error_name) or bool(error_value)
    timeout = "timeout" in f"{error_name or ''} {error_value or ''}".lower()
    normalized_error = "Timeout" if timeout else (str(error_name) if error_name else "ExecutionError" if has_error else None)

    normalized_execution: Dict[str, Any] = {
        **execution_data,
        "logs": logs_data,
        "error": error_data,
    }

    return ExecutionResult(
        exit_code=0 if not has_error else -1,
        stdout=stdout,
        stderr=stderr or str(error_value or ""),
        passed=not has_error,
        error=normalized_error,
        metadata={"execution": normalized_execution},
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
    "get_or_create_sandbox",
    "kill_sandbox",
    "is_timeout_result",
    "run_code_in_sandbox",
    "run_isolated_code",
]
