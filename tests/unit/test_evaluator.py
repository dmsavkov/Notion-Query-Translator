from unittest.mock import AsyncMock, patch

import pytest

from src.evaluator import EvalInputs, Evaluator


@pytest.mark.unit
@pytest.mark.asyncio
async def test_judge_general_returns_stub_payload():
    evaluator = Evaluator()

    result = await evaluator.judge_general(EvalInputs(code="print('x')"))

    assert result["status"] == "stub"
    assert "pending" in result["message"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_eval_context_statements_uses_chat_and_parser():
    evaluator = Evaluator(default_judge_model="gemma27", eval_temperature=0.15, max_tokens=777)
    parsed_rows = [{"statement": "has auth", "status": "present"}]

    with patch("src.evaluator.build_prompt_statements", return_value="PROMPT") as mock_prompt, patch(
        "src.evaluator.async_chat_wrapper",
        new_callable=AsyncMock,
        return_value='[{"statement":"has auth","status":"present"}]',
    ) as mock_chat, patch("src.evaluator.parse_statements_response", return_value=parsed_rows) as mock_parse:
        result = await evaluator.eval_context_statements(
            context="ctx",
            statements=["has auth"],
            judge_model_name="gemma4",
        )

    assert result == parsed_rows
    mock_prompt.assert_called_once_with("ctx", ["has auth"])
    mock_chat.assert_awaited_once()
    kwargs = mock_chat.await_args.kwargs
    assert kwargs["temperature"] == 0.15
    assert kwargs["model_size"] == "gemma4"
    assert kwargs["max_tokens"] == 777
    mock_parse.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_eval_context_statements_returns_empty_when_no_statements():
    evaluator = Evaluator()

    with patch("src.evaluator.async_chat_wrapper", new_callable=AsyncMock) as mock_chat:
        result = await evaluator.eval_context_statements(context="ctx", statements=[])

    assert result == []
    mock_chat.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_eval_code_exec_normalizes_missing_payload():
    evaluator = Evaluator()

    result = await evaluator.eval_code_exec(execution=None, execution_output="fallback output")

    assert result["pass"] is False
    assert result["source"] == "state_missing"
    assert result["output"] == "fallback output"
    assert "Missing precomputed execution result" in result["errors"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_eval_code_exec_extracts_output_and_errors():
    evaluator = Evaluator()

    execution = {
        "passed": False,
        "stdout": "partial stdout",
        "stderr": "traceback line",
        "error": "ValueError: boom",
    }
    result = await evaluator.eval_code_exec(execution=execution)

    assert result["pass"] is False
    assert result["source"] == "state"
    assert result["output"] == "partial stdout"
    assert "traceback line" in result["errors"]
    assert "ValueError: boom" in result["errors"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_eval_code_merges_execution_and_statements_results():
    evaluator = Evaluator()

    with patch.object(
        evaluator,
        "eval_code_exec",
        new_callable=AsyncMock,
        return_value={"pass": True, "output": "ok", "errors": None, "source": "state"},
    ) as mock_exec, patch.object(
        evaluator,
        "eval_context_statements",
        new_callable=AsyncMock,
        return_value=[{"statement": "s1", "status": "present"}],
    ) as mock_stmt:
        result = await evaluator.eval_code(
            code="print('ok')",
            statements=["s1"],
            judge_model_name="gemma4",
            execution_result={"passed": True, "stdout": "ok"},
            execution_output="ok",
        )

    assert result["execution"]["pass"] is True
    assert result["statements"][0]["status"] == "present"
    mock_exec.assert_awaited_once_with(execution={"passed": True, "stdout": "ok"}, execution_output="ok")
    mock_stmt.assert_awaited_once_with(context="print('ok')", statements=["s1"], judge_model_name="gemma4")
