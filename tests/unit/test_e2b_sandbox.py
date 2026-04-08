import json

import pytest
from e2b_code_interpreter import TimeoutException
from unittest.mock import MagicMock, patch
from src.utils.execution_utils import get_or_create_sandbox, kill_sandbox, run_code_in_sandbox


class _DummyLogs:
    def __init__(self, stdout=None, stderr=None):
        self.stdout = stdout or []
        self.stderr = stderr or []


class _DummyError:
    def __init__(self, name: str, value: str):
        self.name = name
        self.value = value


class _DummyExecution:
    def __init__(self, *, payload: dict, logs=None, error=None):
        self._payload = payload
        self.logs = logs or _DummyLogs()
        self.error = error

    def to_json(self) -> str:
        return json.dumps(self._payload)


@pytest.fixture
def mock_sandbox_api():
    with patch("src.utils.execution_utils.Sandbox") as mock_cls:
        instance = MagicMock()
        instance.sandbox_id = "mocked-sandbox-id"
        mock_cls.create.return_value = instance
        mock_cls.connect.return_value = instance
        yield mock_cls, instance


def test_get_or_create_sandbox_creates_new(mock_sandbox_api):
    MockSandbox, instance = mock_sandbox_api
    
    sandbox = get_or_create_sandbox(
        sandbox_id=None,
        thread_id="thread",
        task_id="task",
        template="temp",
        timeout_seconds=60
    )
    
    MockSandbox.create.assert_called_once()
    MockSandbox.connect.assert_not_called()
    assert sandbox == instance


def test_get_or_create_sandbox_connects_existing(mock_sandbox_api):
    MockSandbox, instance = mock_sandbox_api
    
    sandbox = get_or_create_sandbox(
        sandbox_id="existing-id",
        thread_id="thread",
        task_id="task",
        template="temp",
        timeout_seconds=60
    )
    
    MockSandbox.connect.assert_called_once_with("existing-id")
    MockSandbox.create.assert_not_called()
    assert sandbox == instance


def test_get_or_create_sandbox_fallback_on_error(mock_sandbox_api):
    MockSandbox, instance = mock_sandbox_api
    
    # Simulate the sandbox expiring or API erroring on connect
    MockSandbox.connect.side_effect = Exception("Sandbox not found")
    
    sandbox = get_or_create_sandbox(
        sandbox_id="dead-id",
        thread_id="thread",
        task_id="task",
        template="temp",
        timeout_seconds=60
    )
    
    MockSandbox.connect.assert_called_once_with("dead-id")
    MockSandbox.create.assert_called_once()
    assert sandbox == instance


def test_kill_sandbox(mock_sandbox_api):
    MockSandbox, instance = mock_sandbox_api

    kill_sandbox("existing-id")

    MockSandbox.connect.assert_called_once_with("existing-id")
    instance.kill.assert_called_once()


def test_run_code_in_sandbox_parses_stringified_payload_logs_and_error():
    sandbox = MagicMock()
    payload = {
        "logs": json.dumps({"stdout": ["before_err\\n"], "stderr": []}),
        "error": json.dumps({"name": "ZeroDivisionError", "value": "division by zero", "traceback": "tb"}),
    }
    sandbox.run_code.return_value = _DummyExecution(payload=payload)

    result = run_code_in_sandbox(
        sandbox=sandbox,
        code="print('before_err'); 1/0",
        execution_timeout_seconds=10,
    )

    assert result.passed is False
    assert result.exit_code == -1
    assert "before_err" in result.stdout
    assert "division by zero" in result.stderr
    assert result.error == "ZeroDivisionError"


def test_run_code_in_sandbox_falls_back_to_execution_object_fields():
    sandbox = MagicMock()
    payload = {"logs": {}, "error": None}
    execution = _DummyExecution(
        payload=payload,
        logs=_DummyLogs(stdout=["hello\n"], stderr=[]),
        error=_DummyError("ValueError", "bad value"),
    )
    sandbox.run_code.return_value = execution

    result = run_code_in_sandbox(
        sandbox=sandbox,
        code="raise ValueError('bad value')",
        execution_timeout_seconds=10,
    )

    assert result.passed is False
    assert result.stdout == "hello\n"
    assert result.error == "ValueError"
    assert "bad value" in result.stderr


def test_run_code_in_sandbox_normalizes_timeout_exception():
    sandbox = MagicMock()
    sandbox.run_code.side_effect = TimeoutException("Execution timed out")

    result = run_code_in_sandbox(
        sandbox=sandbox,
        code="while True:\n    pass",
        execution_timeout_seconds=2,
    )

    assert result.passed is False
    assert result.error == "Timeout"
    assert "timed out" in result.stderr.lower()
