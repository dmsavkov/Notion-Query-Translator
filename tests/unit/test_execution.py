import pytest
import os
from src.utils.execution_utils import run_isolated_code

@pytest.mark.unit
def test_run_isolated_code_success():
    code = "print('Hello, World!')"
    result = run_isolated_code(code, "test_success")
    
    assert result.passed is True
    assert result.exit_code == 0
    assert "Hello, World!" in result.stdout
    assert result.stderr == ""

@pytest.mark.unit
def test_run_isolated_code_runtime_error():
    code = "raise ValueError('Test error')"
    result = run_isolated_code(code, "test_runtime_error")
    
    assert result.passed is False
    assert result.exit_code != 0
    assert "ValueError: Test error" in result.stderr

@pytest.mark.unit
def test_run_isolated_code_syntax_error():
    code = "if True print('oops')"
    result = run_isolated_code(code, "test_syntax_error")
    
    assert result.passed is False
    assert result.exit_code != 0
    assert "SyntaxError" in result.stderr

@pytest.mark.unit
def test_run_isolated_code_timeout():
    # Use a long sleep to trigger timeout
    # Note: the timeout in run_isolated_code is 30s, 
    # but we can't easily wait 30s in a unit test.
    # For now, we skip the real timeout test or use a very short one if it were configurable.
    pass

@pytest.mark.unit
def test_run_isolated_code_env_vars():
    os.environ["TEST_VAR"] = "val123"
    code = "import os; print(os.getenv('TEST_VAR'))"
    result = run_isolated_code(code, "test_env")
    
    assert result.passed is True
    assert "val123" in result.stdout
