import os

import pytest
from dotenv import load_dotenv

from src.utils.execution_utils import get_or_create_sandbox, run_code_in_sandbox, kill_sandbox

load_dotenv()

# Run with: python -m pytest tests/integration/test_e2b_sandbox_vcr.py --record-mode=once

@pytest.fixture(scope="module")
def api_key_check():
    if not os.getenv("E2B_API_KEY"):
        pytest.skip("E2B_API_KEY is required for VCR integration tests.")


@pytest.mark.vcr(
    record_mode="once",
    ignore_hosts=["api.smith.langchain.com"],
    filter_headers=["authorization", "x-api-key"],
    match_on=["method", "scheme", "host", "port", "path", "query"],
)
def test_run_code_happy_and_error_paths(api_key_check):
    """
    Validates real run_code behavior for a successful and a failing execution.
    """
    task_id = "test_vcr_run_code_paths"

    sandbox = get_or_create_sandbox(
        sandbox_id=None,
        thread_id="test_thread",
        task_id=task_id,
        template="notion-query-execution-sandbox",
        timeout_seconds=60,
    )

    assert sandbox.sandbox_id is not None
    sandbox_id = sandbox.sandbox_id

    try:
        happy_code = "print('success_output')"
        res_happy = run_code_in_sandbox(
            sandbox=sandbox,
            code=happy_code,
            execution_timeout_seconds=10,
        )
        assert res_happy.passed is True
        assert "success_output" in res_happy.stdout

        exc_code = "print('failing'); 1/0"
        res_exc = run_code_in_sandbox(
            sandbox=sandbox,
            code=exc_code,
            execution_timeout_seconds=10,
        )
        assert res_exc.passed is False
        assert res_exc.error == "ZeroDivisionError"
        assert "division by zero" in res_exc.stderr
        assert "failing" in res_exc.stdout
    finally:
        kill_sandbox(sandbox_id)


@pytest.mark.vcr(
    record_mode="once",
    ignore_hosts=["api.smith.langchain.com"],
    filter_headers=["authorization", "x-api-key"],
    match_on=["method", "scheme", "host", "port", "path", "query"],
)
def test_dead_sandbox_fallback(api_key_check):
    """
    Invalid sandbox_id should trigger fallback creation.
    """
    task_id = "test_vcr_connect_fallback"
    bad_id = "sbx-invalid-id-xyz"

    sandbox = get_or_create_sandbox(
        sandbox_id=bad_id,
        thread_id="test_thread_fallback",
        task_id=task_id,
        template="notion-query-execution-sandbox",
        timeout_seconds=60,
    )

    assert sandbox.sandbox_id is not None
    assert sandbox.sandbox_id != bad_id

    kill_sandbox(sandbox.sandbox_id)


@pytest.mark.vcr(
    record_mode="once",
    ignore_hosts=["api.smith.langchain.com"],
    filter_headers=["authorization", "x-api-key"],
    match_on=["method", "scheme", "host", "port", "path", "query"],
)
def test_run_code_timeout_path(api_key_check):
    """Infinite-loop code should normalize to Timeout result."""
    sandbox = get_or_create_sandbox(
        sandbox_id=None,
        thread_id="test_thread_timeout",
        task_id="test_vcr_timeout",
        template="notion-query-execution-sandbox",
        timeout_seconds=60,
    )

    try:
        result = run_code_in_sandbox(
            sandbox=sandbox,
            code="while True:\n    pass",
            execution_timeout_seconds=2,
        )

        assert result.passed is False
        assert result.error == "Timeout"
        assert "timed out" in result.stderr.lower()
    finally:
        kill_sandbox(sandbox.sandbox_id)
