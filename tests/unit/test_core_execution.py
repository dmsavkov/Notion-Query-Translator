from contextlib import AbstractAsyncContextManager
from unittest.mock import AsyncMock, patch

import pytest

from src.adapters.cli_presenter import format_lifecycle_result
from src.core.execute_single import execute_single
from src.core.execute_batch import execute_batch
from src.core.lifecycle import run_with_lifecycle
from src.models.config import AppConfig
from src.models.schema import AgentParams, PipelineParams, RagBuildConfig, StaticParams, generate_default_state


class _DummyAsyncContext(AbstractAsyncContextManager):
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _build_app_config() -> AppConfig:
    return AppConfig(
        pipeline=PipelineParams(),
        static=StaticParams(),
        agent=AgentParams(),
        rag=RagBuildConfig(),
    )


@pytest.mark.unit
def test_generate_default_state_starts_empty():
    state = generate_default_state()

    assert state["task_id"] == ""
    assert state["user_prompt"] == ""
    assert state["meta"] == {}
    assert state["security"] == {}
    assert state["terminal_status"] == "pending"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_single_returns_raw_final_state():
    pipeline = AsyncMock()
    final_state = {
        "task_id": "task-1",
        "user_prompt": "Write code",
        "generated_code": "print(42)",
        "function_name": "main",
        "solution_run": {"exit_code": 0, "stdout": "42\n", "stderr": "", "passed": True},
        "execution_output": "42\n",
        "final_code": "print(42)",
        "trials": [],
    }
    pipeline.ainvoke = AsyncMock(return_value=final_state)

    result = await execute_single(
        tasks={"task-1": {"query": "Write code"}},
        app_config=_build_app_config(),
        pipeline=pipeline,
        thread_id="thread-123",
    )

    assert result == {"task-1": final_state}
    assert result["task-1"]["generated_code"] == "print(42)"
    pipeline.ainvoke.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_single_preserves_additional_task_fields_in_initial_state():
    pipeline = AsyncMock()
    final_state = {
        "task_id": "task-1",
        "user_prompt": "Write code",
        "generated_code": "print(42)",
        "function_name": "main",
        "solution_run": {"exit_code": 0, "stdout": "42\n", "stderr": "", "passed": True},
        "execution_output": "42\n",
        "final_code": "print(42)",
        "trials": [],
    }
    pipeline.ainvoke = AsyncMock(return_value=final_state)

    result = await execute_single(
        tasks={"task-1": {"query": "Write code", "think": True, "notes": "preserve me"}},
        app_config=_build_app_config(),
        pipeline=pipeline,
        thread_id="thread-123",
    )

    assert result == {"task-1": final_state}
    initial_state = pipeline.ainvoke.await_args.args[0]
    assert initial_state["task_id"] == "task-1"
    assert initial_state["user_prompt"] == "Write code"
    assert initial_state["think"] is True
    assert initial_state["notes"] == "preserve me"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_single_reads_prompt_from_nested_input_state():
    pipeline = AsyncMock()
    final_state = {
        "task_id": "task-1",
        "user_prompt": "Nested prompt",
        "generated_code": "print(42)",
        "function_name": "main",
        "solution_run": {"exit_code": 0, "stdout": "42\n", "stderr": "", "passed": True},
        "execution_output": "42\n",
        "final_code": "print(42)",
        "trials": [],
    }
    pipeline.ainvoke = AsyncMock(return_value=final_state)

    result = await execute_single(
        tasks={
            "task-1": {
                "input_state": {"user_prompt": "Nested prompt", "extra": "value"},
                "reference_outputs": {"relevant_to_notion_scope": True},
            }
        },
        app_config=_build_app_config(),
        pipeline=pipeline,
        thread_id="thread-123",
    )

    assert result == {"task-1": final_state}
    initial_state = pipeline.ainvoke.await_args.args[0]
    assert initial_state["task_id"] == "task-1"
    assert initial_state["user_prompt"] == "Nested prompt"
    assert initial_state["input_state"]["extra"] == "value"
    assert initial_state["reference_outputs"]["relevant_to_notion_scope"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_single_returns_error_state_when_pipeline_raises():
    pipeline = AsyncMock()
    pipeline.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

    result = await execute_single(
        tasks={"task-1": {"query": "Write code"}},
        app_config=_build_app_config(),
        pipeline=pipeline,
        thread_id="thread-123",
    )

    task_result = result["task-1"]
    assert task_result["task_id"] == "task-1"
    assert task_result["user_prompt"] == "Write code"
    assert task_result["generated_code"] == ""
    assert task_result["solution_run"] == {}
    assert task_result["error"] == "boom"


@pytest.mark.unit
def test_format_lifecycle_result_uses_generated_code_fallback():
    rendered = format_lifecycle_result(
        {
            "task-1": {
                "solution_run": {"passed": True},
                "execution_output": "ok\n",
                "generated_code": "print(42)",
            }
        }
    )

    assert "[task-1] PASS" in rendered
    assert "Execution Output:" in rendered
    assert "ok" in rendered
    assert "Generated Code:" in rendered
    assert "print(42)" in rendered


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_with_lifecycle_calls_single_task_executor_directly():
    app_config = _build_app_config()
    tasks = {"task-1": {"query": "Write code"}}

    with patch("src.core.lifecycle.AsyncSqliteSaver.from_conn_string", return_value=_DummyAsyncContext()), patch(
        "src.core.lifecycle.build_pipeline"
    ) as mock_build_pipeline, patch(
        "src.core.lifecycle.execute_single",
        new_callable=AsyncMock,
        return_value={
            "task-1": {
                "task_id": "task-1",
                "generated_code": "print(42)",
                "solution_run": {"passed": True},
                "execution_output": "42\n",
            }
        },
    ) as mock_execute_single:
        mock_build_pipeline.return_value.compile.return_value = object()

        result = await run_with_lifecycle(tasks=tasks, app_config=app_config)

    assert result["task-1"]["generated_code"] == "print(42)"
    mock_execute_single.assert_awaited_once()
    assert mock_execute_single.await_args is not None
    assert mock_execute_single.await_args.kwargs["tasks"] == tasks


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_batch_routes_through_execute_single():
    app_config = _build_app_config()
    tasks = {
        "task-1": {"query": "Write code"},
        "task-2": {"query": "Write more code"},
    }

    with patch("src.core.execute_batch.execute_single", new_callable=AsyncMock) as mock_execute_single, patch(
        "src.core.execute_batch.generate_thread_id",
        side_effect=["thread-1", "thread-2"],
    ) as mock_thread_id:
        mock_execute_single.side_effect = [
            {"task-1": {"task_id": "task-1", "generated_code": "print(1)"}},
            {"task-2": {"task_id": "task-2", "generated_code": "print(2)"}},
        ]

        result = await execute_batch(tasks=tasks, app_config=app_config, pipeline=object())

    assert result["task-1"]["generated_code"] == "print(1)"
    assert result["task-2"]["generated_code"] == "print(2)"
    assert mock_execute_single.await_count == 2
    assert mock_execute_single.await_args_list[0].kwargs["tasks"] == {"task-1": tasks["task-1"]}
    assert mock_execute_single.await_args_list[1].kwargs["tasks"] == {"task-2": tasks["task-2"]}
    assert mock_thread_id.call_args_list[0].args == ("task-1",)
    assert mock_thread_id.call_args_list[1].args == ("task-2",)
