import asyncio
from typing import Any, Callable, Dict, List, Literal, Optional

from langsmith import Client
from langsmith.evaluation import aevaluate
from pydantic import BaseModel, ConfigDict, computed_field

from .all_functionality import load_eval_tasks


class StandardEvaluationSettings(BaseModel):
    """Standard, shared configuration for LangSmith-based evaluations."""

    experiment_prefix: str = "E2E Evaluation"
    evals_case_type: Literal["simple", "complex", "all"] = "complex"
    eval_max_concurrency: int = 5
    run_error_analysis_after_eval: bool = True
    evals_dir: str = "evals"
    provision_infrastructure: bool = True

    model_config = ConfigDict(frozen=True)

    @computed_field
    @property
    def dataset_name(self) -> str:
        mapping = {
            "complex": "Dataset v4.",
            "simple": "Dataset v1.",
            "all": "Dataset v3.",
        }
        return mapping[self.evals_case_type]


EvaluationSettings = StandardEvaluationSettings


def extract_task_prompt(task_spec: Dict[str, Any]) -> str:
    return str(task_spec.get("query") or task_spec.get("user_prompt") or task_spec.get("task") or "").strip()


def build_reference_outputs(task_id: str, task_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Reference outputs attached to dataset examples (ground truth only)."""
    task = extract_task_prompt(task_spec)
    return {
        "task_id": task_id,
        "task": task,
        "solution": task_spec.get("solution", ""),
        "correct_statements": task_spec.get("correct_statements", []) or [],
    }


def load_eval_tasks_or_raise(settings: StandardEvaluationSettings) -> Dict[str, Dict[str, Any]]:
    task_specs = load_eval_tasks(settings.evals_dir, case_type=settings.evals_case_type)
    if not task_specs:
        raise ValueError(
            f"No evaluation tasks loaded for case_type='{settings.evals_case_type}'. "
            "Check evals directory and case filters."
        )
    return task_specs


def ensure_dataset(client: Client, dataset_name: str, task_specs: Dict[str, Dict[str, Any]]) -> Any:
    """
    Ensure a stable dataset exists and contains one example per task.
    Existing examples are reused; only missing examples are created.
    """
    dataset = None
    for ds in client.list_datasets(dataset_name=dataset_name):
        if ds.name == dataset_name:
            dataset = ds
            break

    if dataset is None:
        dataset = client.create_dataset(dataset_name=dataset_name)
        print(f"Created dataset: {dataset_name}")
    else:
        print(f"Using existing dataset: {dataset_name}")

    existing_by_task_id: Dict[str, Any] = {}
    for ex in client.list_examples(dataset_id=dataset.id):
        ex_inputs = getattr(ex, "inputs", {}) or {}
        task_id = str(ex_inputs.get("task_id") or "").strip()
        if task_id:
            existing_by_task_id[task_id] = ex

    created = 0
    for task_id, task_spec in sorted(task_specs.items()):
        if task_id in existing_by_task_id:
            continue
        client.create_example(
            inputs={
                "task_id": task_id,
                "query": extract_task_prompt(task_spec),
            },
            outputs=build_reference_outputs(task_id, task_spec),
            dataset_id=dataset.id,
        )
        created += 1

    print(f"Dataset '{dataset_name}' ready. Existing: {len(existing_by_task_id)} | Created now: {created}")
    return dataset


def _extract_execution_error(result: Any) -> Optional[Dict[str, str]]:
    """Extract execution errors from aevaluate iterator records."""
    if isinstance(result, dict):
        top_error = result.get("error")
        run_payload = result.get("run") or {}
        run_name = str(run_payload.get("name") or result.get("example_id") or result.get("id") or "Unknown")
        outputs = run_payload.get("outputs") if isinstance(run_payload, dict) else {}
    else:
        top_error = getattr(result, "error", None)
        run_payload = getattr(result, "run", None)
        run_name = str(
            getattr(run_payload, "name", None)
            or getattr(result, "example_id", None)
            or getattr(result, "id", None)
            or "Unknown"
        )
        outputs = getattr(run_payload, "outputs", {}) if run_payload is not None else {}

    if top_error:
        return {"task": run_name, "error": str(top_error)}

    if isinstance(outputs, dict):
        output_error = outputs.get("error")
        if output_error:
            return {"task": run_name, "error": str(output_error)}

    return None


async def evaluation_orchestration(
    *,
    settings: StandardEvaluationSettings,
    target: Callable[[Dict[str, Any]], Any],
    evaluators: List[Any],
    human_config: Optional[Any] = None,
    client: Optional[Client] = None,
) -> Dict[str, Any]:
    """
    Shared orchestration for evaluation scripts/tests.

    Assertion policy is intentionally strict and narrow:
    fail only when LangSmith execution records contain runtime errors/exceptions.
    """
    eval_client = client or Client()

    task_specs = load_eval_tasks_or_raise(settings)

    if settings.provision_infrastructure:
        from src.evaluation.sandbox import provision_infrastructure

        provision_infrastructure()

    ensure_dataset(eval_client, settings.dataset_name, task_specs)

    print(f"\\n[Executing Eval] Dataset: {settings.dataset_name} | Prefix: {settings.experiment_prefix}")

    results_iterator = await aevaluate(
        target,
        data=settings.dataset_name,
        evaluators=evaluators,
        experiment_prefix=settings.experiment_prefix,
        max_concurrency=settings.eval_max_concurrency,
    )

    failed_executions: List[Dict[str, str]] = []
    async for result in results_iterator:
        error_record = _extract_execution_error(result)
        if error_record:
            failed_executions.append(error_record)

    if failed_executions:
        lines = [
            f"Evaluation {settings.experiment_prefix} had {len(failed_executions)} execution crashes:",
            *[f" - Task {item['task']}: {item['error']}" for item in failed_executions],
        ]
        raise AssertionError("\n".join(lines))

    error_analysis_result: Optional[Dict[str, Any]] = None
    if settings.run_error_analysis_after_eval:
        from src.error_analysis import HumanConfig, main as run_error_analysis_main

        error_analysis_result = await asyncio.to_thread(
            run_error_analysis_main,
            settings.experiment_prefix,
            human_config or HumanConfig(),
            settings.dataset_name,
        )

    return {
        "failed_executions": failed_executions,
        "error_analysis_result": error_analysis_result,
    }
