from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation.utils import (
    StandardEvaluationSettings,
    _extract_execution_error,
    build_reference_outputs,
    ensure_dataset,
    evaluation_orchestration,
    extract_task_prompt,
    load_eval_tasks_or_raise,
)


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


@pytest.mark.unit
def test_standard_evaluation_settings_dataset_name_mapping():
    assert StandardEvaluationSettings(evals_case_type="complex").dataset_name == "Dataset v4."
    assert StandardEvaluationSettings(evals_case_type="simple").dataset_name == "Dataset v1."
    assert StandardEvaluationSettings(evals_case_type="all").dataset_name == "Dataset v3."


@pytest.mark.unit
def test_extract_task_prompt_uses_expected_fallback_order():
    assert extract_task_prompt({"query": "  use query  ", "user_prompt": "ignored"}) == "use query"
    assert extract_task_prompt({"user_prompt": "prompt"}) == "prompt"
    assert extract_task_prompt({"task": "task text"}) == "task text"
    assert extract_task_prompt({"query": "", "user_prompt": "", "task": ""}) == ""


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
    existing_example = SimpleNamespace(inputs={"task_id": "task_1"})

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
    client.create_example.assert_called_once()
    kwargs = client.create_example.call_args.kwargs
    assert kwargs["dataset_id"] == "ds_1"
    assert kwargs["inputs"] == {"task_id": "task_2", "query": "create me"}
    assert kwargs["outputs"]["task_id"] == "task_2"


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

    with patch("src.evaluation.utils.load_eval_tasks_or_raise", return_value={"task-x": {"query": "q"}}), patch(
        "src.evaluation.utils.ensure_dataset"
    ) as mock_ensure, patch("src.evaluation.utils.aevaluate", new_callable=AsyncMock, return_value=_AsyncIterator(rows)):
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

    with patch("src.evaluation.utils.load_eval_tasks_or_raise", return_value={"task-1": {"query": "q"}}), patch(
        "src.evaluation.utils.ensure_dataset"
    ), patch("src.evaluation.utils.aevaluate", new_callable=AsyncMock, return_value=_AsyncIterator([])), patch(
        "src.evaluation.sandbox.provision_infrastructure"
    ) as mock_provision, patch("src.evaluation.utils.asyncio.to_thread", new_callable=AsyncMock, return_value={"ok": True}) as mock_to_thread:
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
