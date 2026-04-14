from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest
import yaml

from src.evaluation.shared import (
    ExactMatchEvaluator,
    evaluation_orchestration,
    make_live_eval_target,
)
from src.evaluation.utils import (
    StandardEvaluationSettings,
    _extract_execution_error,
    build_node_eval_state,
    build_reference_outputs,
    ensure_dataset,
    extract_task_prompt,
    load_eval_tasks_or_raise,
)
from src.models.schema import PipelineParams, StaticParams


class _AsyncIterator:
    def __init__(self, rows):
        self._rows = list(rows)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._rows):
            raise StopAsyncIteration
        value = self._rows[self._index]
        self._index += 1
        return value


def _noop_openai_client_session(*_a, **_k):
    @asynccontextmanager
    async def _cm():
        yield None

    return _cm()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exact_match_evaluator_compares_only_available_reference_keys():
    evaluator = ExactMatchEvaluator(keys_to_check=["is_safe", "extra_metadata"], metric_key="exact_test")

    result = await evaluator(
        inputs={"task_id": "task-1"},
        outputs={"task_id": "task-1", "is_safe": True, "extra_metadata": "present"},
        reference_outputs={"task_id": "task-1", "is_safe": True},
    )

    assert result["key"] == "exact_test"
    assert result["score"] == 1.0
    details = json.loads(result["comment"])
    assert details["task_id"] == "task-1"
    assert details["checked"] == 1
    assert details["matched"] == 1


@pytest.mark.unit
def test_build_node_eval_state_uses_nested_input_state():
    state = build_node_eval_state(
        {
            "task_id": "task-1",
            "input_state": {
                "user_prompt": "Nested prompt",
                "notes": "keep",
            },
            "reference_outputs": {"is_safe": True},
        }
    )

    assert state["task_id"] == "task-1"
    assert state["user_prompt"] == "Nested prompt"
    assert state["notes"] == "keep"


@pytest.mark.unit
def test_build_node_eval_state_uses_flat_inputs_when_input_state_missing():
    state = build_node_eval_state({"task_id": "task-2", "query": "Flat prompt", "flag": True})

    assert state["task_id"] == "task-2"
    assert state["user_prompt"] == "Flat prompt"
    assert state["flag"] is True


@pytest.mark.unit
def test_standard_evaluation_settings_uses_explicit_dataset_name():
    assert StandardEvaluationSettings().dataset_name == "Dataset v4."
    assert StandardEvaluationSettings(evals_case_type="simple").dataset_name == "Dataset v4."
    assert StandardEvaluationSettings(dataset_name="Dataset custom.").dataset_name == "Dataset custom."


@pytest.mark.unit
def test_extract_task_prompt_uses_expected_fallback_order():
    assert extract_task_prompt({"query": "  use query  ", "user_prompt": "ignored"}) == "use query"
    assert extract_task_prompt({"user_prompt": "prompt"}) == "prompt"
    assert extract_task_prompt({"task": "task text"}) == "task text"
    assert extract_task_prompt({"query": "", "user_prompt": "", "task": ""}) == ""
    assert extract_task_prompt({"input_state": {"user_prompt": "nested prompt"}}) == "nested prompt"


@pytest.mark.unit
def test_build_reference_outputs_includes_solution_and_statements_defaults():
    out = build_reference_outputs(
        "task_a",
        {
            "query": "write code",
            "solution": "print(1)",
            "correct_statements": ["uses print"],
        },
    )

    assert out["task_id"] == "task_a"
    assert out["task"] == "write code"
    assert out["solution"] == "print(1)"
    assert out["correct_statements"] == ["uses print"]

    defaults = build_reference_outputs("task_b", {"task": "do work"})
    assert defaults["solution"] == ""
    assert defaults["correct_statements"] == []

    nested = build_reference_outputs(
        "task_c",
        {
            "input_state": {"user_prompt": "nested request"},
            "reference_outputs": {
                "relevant_to_notion_scope": True,
            },
        },
    )
    assert nested["task"] == "nested request"
    assert nested["relevant_to_notion_scope"] is True


@pytest.mark.unit
def test_load_eval_tasks_or_raise_raises_when_no_tasks_loaded():
    settings = StandardEvaluationSettings(evals_case_type="complex")

    with patch("src.evaluation.utils.load_eval_tasks", return_value={}):
        with pytest.raises(ValueError, match="No evaluation tasks loaded"):
            load_eval_tasks_or_raise(settings)


@pytest.mark.unit
def test_load_eval_tasks_or_raise_returns_loaded_tasks():
    settings = StandardEvaluationSettings(evals_case_type="simple")
    tasks = {"task_1": {"query": "hello"}}

    with patch("src.evaluation.utils.load_eval_tasks", return_value=tasks) as mock_loader:
        result = load_eval_tasks_or_raise(settings)

    assert result == tasks
    mock_loader.assert_called_once_with(settings.evals_dir, case_type="simple")


@pytest.mark.unit
def test_ensure_dataset_creates_missing_examples_only():
    dataset = SimpleNamespace(id="ds_1", name="Dataset v4.")
    existing_example = SimpleNamespace(id="ex_existing", inputs={"task_id": "task_1", "query": "already there"})

    client = MagicMock()
    client.list_datasets.return_value = [dataset]
    client.list_examples.return_value = [existing_example]

    task_specs = {
        "task_1": {"query": "already there", "solution": "x"},
        "task_2": {"user_prompt": "create me", "solution": "y", "correct_statements": ["s"]},
    }

    out_dataset = ensure_dataset(client, "Dataset v4.", task_specs)

    assert out_dataset is dataset
    client.create_dataset.assert_not_called()
    client.update_example.assert_not_called()
    client.create_example.assert_called_once()
    kwargs = client.create_example.call_args.kwargs
    assert kwargs["dataset_id"] == "ds_1"
    assert kwargs["inputs"] == {"task_id": "task_2", "query": "create me"}
    assert kwargs["outputs"]["task_id"] == "task_2"


@pytest.mark.unit
def test_ensure_dataset_flattens_input_state_into_langsmith_inputs():
    dataset = SimpleNamespace(id="ds_2", name="Dataset flat")

    client = MagicMock()
    client.list_datasets.return_value = [dataset]
    client.list_examples.return_value = []

    task_specs = {
        "task_nested": {
            "input_state": {
                "user_prompt": "nested prompt",
                "custom_flag": True,
            },
            "reference_outputs": {"is_safe": True},
        }
    }

    out_dataset = ensure_dataset(client, "Dataset flat", task_specs)

    assert out_dataset is dataset
    client.create_example.assert_called_once()
    kwargs = client.create_example.call_args.kwargs
    assert kwargs["inputs"] == {
        "task_id": "task_nested",
        "user_prompt": "nested prompt",
        "custom_flag": True,
    }


@pytest.mark.unit
def test_ensure_dataset_updates_existing_examples_when_inputs_change():
    dataset = SimpleNamespace(id="ds_3", name="Dataset sync")
    existing_example = SimpleNamespace(
        id="ex_1",
        inputs={"task_id": "task_sync", "user_prompt": "old prompt"},
    )

    client = MagicMock()
    client.list_datasets.return_value = [dataset]
    client.list_examples.return_value = [existing_example]

    task_specs = {
        "task_sync": {
            "input_state": {
                "user_prompt": "new prompt",
                "required_resources": ["AI Research"],
            },
        }
    }

    out_dataset = ensure_dataset(client, "Dataset sync", task_specs)

    assert out_dataset is dataset
    client.create_example.assert_not_called()
    client.update_example.assert_called_once()
    call_args = client.update_example.call_args
    assert call_args.args[0] == "ex_1"
    assert call_args.kwargs["inputs"] == {
        "task_id": "task_sync",
        "user_prompt": "new prompt",
        "required_resources": ["AI Research"],
    }


@pytest.mark.unit
def test_load_eval_tasks_or_raise_supports_selector_file_path(tmp_path):
    evals_root = tmp_path / "evals"
    precheck_dir = evals_root / "precheck"
    precheck_dir.mkdir(parents=True)
    (precheck_dir / "general_precheck_v1.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "input_state": {"user_prompt": "Selector prompt"},
                    "reference_outputs": {
                        "relevant_to_notion_scope": True,
                    },
                }
            ],
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    settings = StandardEvaluationSettings(
        evals_dir=str(evals_root),
        evals_case_type="precheck/general_precheck_v1.yaml",
        provision_infrastructure=False,
    )

    tasks = load_eval_tasks_or_raise(settings)
    assert list(tasks.keys()) == ["general_precheck_v1__01"]


@pytest.mark.unit
def test_extract_execution_error_handles_dict_and_object_shapes():
    dict_shape = {
        "run": {"name": "task-1", "outputs": {"error": "runtime boom"}},
        "error": None,
    }
    assert _extract_execution_error(dict_shape) == {"task": "task-1", "error": "runtime boom"}

    obj_shape = SimpleNamespace(
        error="top-level crash",
        run=SimpleNamespace(name="task-2", outputs={}),
    )
    assert _extract_execution_error(obj_shape) == {"task": "task-2", "error": "top-level crash"}


@pytest.mark.unit
def test_ensure_dataset_fails_fast_when_langsmith_key_missing(monkeypatch: pytest.MonkeyPatch):
    dataset = SimpleNamespace(id="ds_auth", name="Dataset auth")

    client = MagicMock()
    client.list_datasets.return_value = [dataset]
    client.list_examples.return_value = []

    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    with patch("src.evaluation.utils.load_runtime_environment") as mock_load_env:
        with pytest.raises(EnvironmentError, match="LangSmith auth preflight failed"):
            ensure_dataset(client, "Dataset auth", {"task_1": {"query": "q"}})

    mock_load_env.assert_called_once_with(include_sandbox=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evaluation_orchestration_raises_on_execution_errors():
    settings = StandardEvaluationSettings(
        experiment_prefix="unit-test-eval",
        evals_case_type="simple",
        run_error_analysis_after_eval=False,
        provision_infrastructure=False,
    )

    async def target(_inputs):
        return {}

    rows = [{"run": {"name": "task-x", "outputs": {"error": "boom"}}}]
    client = MagicMock()
    client.list_datasets.return_value = []

    with patch("src.evaluation.shared.load_eval_tasks_or_raise", return_value={"task-x": {"query": "q"}}), patch(
        "src.evaluation.shared.ensure_dataset"
    ) as mock_ensure, patch(
        "src.evaluation.shared.openai_client_session",
        side_effect=_noop_openai_client_session,
    ), patch("src.evaluation.shared.aevaluate", new_callable=AsyncMock, return_value=_AsyncIterator(rows)):
        with pytest.raises(AssertionError, match="task-x"):
            await evaluation_orchestration(
                settings=settings,
                target=target,
                evaluators=[],
                client=client,
            )

    mock_ensure.assert_called_once_with(client, settings.dataset_name, {"task-x": {"query": "q"}})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evaluation_orchestration_waits_after_dataset_sync_when_configured():
    settings = StandardEvaluationSettings(
        experiment_prefix="unit-test-eval",
        evals_case_type="simple",
        run_error_analysis_after_eval=False,
        provision_infrastructure=False,
        post_dataset_sync_delay_seconds=30,
    )

    async def target(_inputs):
        return {}

    client = MagicMock()
    client.list_datasets.return_value = []

    with patch("src.evaluation.shared.load_eval_tasks_or_raise", return_value={"task-x": {"query": "q"}}), patch(
        "src.evaluation.shared.ensure_dataset"
    ) as mock_ensure, patch(
        "src.evaluation.shared.asyncio.sleep",
        new_callable=AsyncMock,
        return_value=None,
    ) as mock_sleep, patch(
        "src.evaluation.shared.openai_client_session",
        side_effect=_noop_openai_client_session,
    ), patch("src.evaluation.shared.aevaluate", new_callable=AsyncMock, return_value=_AsyncIterator([])):
        result = await evaluation_orchestration(
            settings=settings,
            target=target,
            evaluators=[],
            client=client,
        )

    assert result == {"failed_executions": [], "error_analysis_result": None}
    mock_ensure.assert_called_once_with(client, settings.dataset_name, {"task-x": {"query": "q"}})
    mock_sleep.assert_awaited_once_with(30.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evaluation_orchestration_runs_provisioning_and_error_analysis_when_enabled():
    settings = StandardEvaluationSettings(
        experiment_prefix="unit-test-eval",
        evals_case_type="complex",
        run_error_analysis_after_eval=True,
        provision_infrastructure=True,
    )
    human_cfg = object()

    async def target(_inputs):
        return {}

    client = MagicMock()
    client.list_datasets.return_value = []

    with patch("src.evaluation.shared.load_eval_tasks_or_raise", return_value={"task-1": {"query": "q"}}), patch(
        "src.evaluation.shared.ensure_dataset"
    ), patch(
        "src.evaluation.shared.openai_client_session",
        side_effect=_noop_openai_client_session,
    ), patch("src.evaluation.shared.aevaluate", new_callable=AsyncMock, return_value=_AsyncIterator([])), patch(
        "src.evaluation.sandbox.provision_infrastructure"
    ) as mock_provision, patch("src.evaluation.shared.asyncio.to_thread", new_callable=AsyncMock, return_value={"ok": True}) as mock_to_thread:
        result = await evaluation_orchestration(
            settings=settings,
            target=target,
            evaluators=[],
            human_config=human_cfg,
            client=client,
        )

    assert result == {"failed_executions": [], "error_analysis_result": {"ok": True}}
    mock_provision.assert_called_once_with()
    assert mock_to_thread.await_args is not None
    assert mock_to_thread.await_args.args[1] == settings.experiment_prefix
    assert mock_to_thread.await_args.args[2] is human_cfg
    assert mock_to_thread.await_args.args[3] == settings.dataset_name


@pytest.mark.unit
@pytest.mark.asyncio
async def test_live_eval_target_uses_lifecycle_and_maps_core_fields():
    target = make_live_eval_target(
        StaticParams(context_used="baseline"),
        PipelineParams(minimal=False),
    )

    final_state = {
        "task_id": "task-1",
        "retrieval_context": "ctx",
        "generated_code": "print(42)",
        "final_code": "",
        "solution_run": {"passed": True, "stdout": "42\n", "stderr": "", "exit_code": 0},
        "execution_output": "42\n",
        "function_name": "main",
    }

    with patch("src.evaluation.shared.run_with_lifecycle", new_callable=AsyncMock, return_value={"task-1": final_state}) as mock_lifecycle:
        out = await target({"task_id": "task-1", "query": "Write code"})

    assert out["task_id"] == "task-1"
    assert out["retrieval_context"] == "ctx"
    assert out["final_code"] == "print(42)"
    assert out["execution"] == final_state["solution_run"]
    assert out["solution_run"] == final_state["solution_run"]
    assert out["execution_output"] == "42\n"
    assert out["error"] == ""
    mock_lifecycle.assert_awaited_once()
    assert mock_lifecycle.await_args is not None
    assert mock_lifecycle.await_args.kwargs["tasks"] == {"task-1": {"query": "Write code"}}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_live_eval_target_prefers_input_state_payload_when_present():
    target = make_live_eval_target(
        StaticParams(context_used="baseline"),
        PipelineParams(minimal=False),
    )

    input_state = {
        "query": "Write code",
        "think": True,
        "notes": "preserve me",
    }
    final_state = {
        "task_id": "task-2",
        "retrieval_context": "ctx-2",
        "generated_code": "print(7)",
        "final_code": "",
        "solution_run": {"passed": True, "stdout": "7\n", "stderr": "", "exit_code": 0},
        "execution_output": "7\n",
        "function_name": "main",
    }

    with patch("src.evaluation.shared.run_with_lifecycle", new_callable=AsyncMock, return_value={"task-2": final_state}) as mock_lifecycle:
        out = await target({"task_id": "task-2", "input_state": input_state})

    assert out["task_id"] == "task-2"
    assert out["retrieval_context"] == "ctx-2"
    assert out["final_code"] == "print(7)"
    assert out["error"] == ""
    mock_lifecycle.assert_awaited_once()
    assert mock_lifecycle.await_args is not None
    assert mock_lifecycle.await_args.kwargs["tasks"] == {"task-2": input_state}
