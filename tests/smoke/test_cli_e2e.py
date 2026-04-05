import os
import shutil
import subprocess
import sys

import pytest


@pytest.mark.smoke
@pytest.mark.llm
def test_cli_run_pipeline_e2e_real_llm_no_patching():
    """End-to-end smoke test: CLI -> pipeline -> real LLM call."""
    cli_exe = shutil.which("notion-agent")
    cmd = [cli_exe] if cli_exe else [sys.executable, "-m", "notion_query.cli"]

    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY is required for real LLM smoke test")

    cmd.extend([
        "Add a new task called 'Pay invoices' to the Notion task database and set its status to 'In Progress'.",
    ])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=420,
        check=False,
    )

    stderr = result.stderr or ""
    if "already accessed by another instance of Qdrant client" in stderr:
        pytest.skip("Qdrant local storage is locked by another running process")

    # Process-level execution checks
    assert result.returncode == 0, (
        f"CLI failed with exit code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    # Real E2E may emit warnings/log lines, but should not fail with traceback/errors.
    assert "Traceback" not in stderr, f"Unexpected traceback in stderr:\n{stderr}"
    assert "UnicodeEncodeError" not in stderr, f"Encoding error in stderr:\n{stderr}"
    assert "Error:" not in stderr, f"Fatal CLI error in stderr:\n{stderr}"

    # Validate pipeline resolved and printed final code section
    assert "Generated Code:" in result.stdout, f"Missing generated code output:\n{result.stdout}"
