import os
from typing import Any, Dict, Optional

from langsmith import Client
from pydantic import BaseModel, ConfigDict

from ..all_functionality import load_eval_tasks
from ..models.schema import generate_default_state
from notion_query.environment import load_runtime_environment


class StandardEvaluationSettings(BaseModel):
    """Standard, shared configuration for LangSmith-based evaluations."""

    experiment_prefix: str = "E2E Evaluation"
    dataset_name: str = "Dataset v4."
    evals_case_type: str = "complex"
    eval_max_concurrency: int = 5
    run_error_analysis_after_eval: bool = True
    evals_dir: str = "evals"
    provision_infrastructure: bool = True
    post_dataset_sync_delay_seconds: float = 0.0

    model_config = ConfigDict(frozen=True)


EvaluationSettings = StandardEvaluationSettings


def _ensure_langsmith_api_key() -> None:
    load_runtime_environment(include_sandbox=True)
    if not ((os.getenv("LANGSMITH_API_KEY") or "").strip() or (os.getenv("LANGCHAIN_API_KEY") or "").strip()):
        raise EnvironmentError("LangSmith auth preflight failed: missing API key.")


def extract_task_prompt(task_spec: Dict[str, Any]) -> str:
    input_state = task_spec.get("input_state")
    if isinstance(input_state, dict):
        nested_prompt = str(
            input_state.get("query") or input_state.get("user_prompt") or input_state.get("task") or ""
        ).strip()
        if nested_prompt:
            return nested_prompt

    return str(task_spec.get("query") or task_spec.get("user_prompt") or task_spec.get("task") or "").strip()


def build_reference_outputs(task_id: str, task_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Reference outputs attached to dataset examples (ground truth only)."""
    task = extract_task_prompt(task_spec)
    solution = task_spec.get("solution", "")
    correct_statements = task_spec.get("correct_statements", []) or []

    nested_reference_outputs = task_spec.get("reference_outputs")
    if isinstance(nested_reference_outputs, dict):
        if not solution:
            solution = nested_reference_outputs.get("solution", "")
        if not correct_statements:
            correct_statements = nested_reference_outputs.get("correct_statements", []) or []

    reference_outputs = {
        "task_id": task_id,
        "task": task,
        "solution": solution,
        "correct_statements": correct_statements,
    }

    if isinstance(nested_reference_outputs, dict):
        for key, value in nested_reference_outputs.items():
            if key not in {"task_id", "task", "solution", "correct_statements"}:
                reference_outputs[key] = value

    return reference_outputs


def build_node_eval_state(inputs: Dict[str, Any], *, default_task_id: str = "") -> Dict[str, Any]:
    """Build a node-ready state dict from LangSmith inputs.

    This normalizes either a nested `input_state` payload or flat key-value
    inputs into the standard graph state used by node-level evaluations.
    """
    state = dict(generate_default_state())

    input_state = inputs.get("input_state")
    if isinstance(input_state, dict):
        state.update(dict(input_state))
    else:
        state.update({k: v for k, v in inputs.items() if k != "reference_outputs"})

    task_id = str(inputs.get("task_id") or state.get("task_id") or default_task_id).strip() or default_task_id
    if task_id:
        state["task_id"] = task_id

    user_prompt = str(
        state.get("user_prompt")
        or state.get("query")
        or state.get("task")
        or inputs.get("user_prompt")
        or inputs.get("query")
        or inputs.get("task")
        or ""
    ).strip()
    if user_prompt:
        state["user_prompt"] = user_prompt

    return state


def _get_value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _extract_execution_error(result: Any) -> Optional[Dict[str, str]]:
    """Extract execution errors from aevaluate iterator records."""
    top_error = _get_value(result, "error")
    run_payload = _get_value(result, "run")
    run_name = str(
        _get_value(run_payload, "name")
        or _get_value(result, "example_id")
        or _get_value(result, "id")
        or "Unknown"
    )
    outputs = _get_value(run_payload, "outputs", {}) if run_payload is not None else {}

    if top_error:
        return {"task": run_name, "error": str(top_error)}

    output_error = _get_value(outputs, "error")
    if output_error:
        return {"task": run_name, "error": str(output_error)}

    return None


def load_eval_tasks_or_raise(settings: StandardEvaluationSettings) -> Dict[str, Dict[str, Any]]:
    task_specs = load_eval_tasks(settings.evals_dir, case_type=settings.evals_case_type)
    if not task_specs:
        raise ValueError(
            f"No evaluation tasks loaded for selector='{settings.evals_case_type}'. "
            "Check evals root, file path, folder name, and selector filters."
        )
    return task_specs


def ensure_dataset(client: Client, dataset_name: str, task_specs: Dict[str, Dict[str, Any]]) -> Any:
    """
    Ensure a stable dataset exists and contains one example per task.
    Existing examples are updated when input payloads drift.
    """
    _ensure_langsmith_api_key()

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
    updated = 0
    for task_id, task_spec in sorted(task_specs.items()):
        inputs = {
            "task_id": task_id,
        }
        input_state = task_spec.get("input_state")
        if isinstance(input_state, dict):
            for key, value in input_state.items():
                if key != "task_id":
                    inputs[key] = value

        prompt = extract_task_prompt(task_spec)
        if prompt and not any(str(inputs.get(k) or "").strip() for k in ("query", "user_prompt", "task")):
            inputs["query"] = prompt

        reference_outputs = build_reference_outputs(task_id, task_spec)

        existing_example = existing_by_task_id.get(task_id)
        if existing_example is not None:
            existing_inputs = getattr(existing_example, "inputs", {}) or {}
            if existing_inputs == inputs:
                continue

            example_id = str(getattr(existing_example, "id", "") or "").strip()
            if not example_id:
                raise ValueError(f"Dataset example for task_id='{task_id}' has no id; cannot update inputs.")

            client.update_example(
                example_id,
                inputs=inputs,
                outputs=reference_outputs,
            )
            updated += 1
            continue

        client.create_example(
            inputs=inputs,
            outputs=reference_outputs,
            dataset_id=dataset.id,
        )
        created += 1

    print(
        f"Dataset '{dataset_name}' ready. "
        f"Existing: {len(existing_by_task_id)} | Updated now: {updated} | Created now: {created}"
    )
    return dataset


def _synthesize_eval_context(
    *,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Synthesize all evaluation context from LangSmith parameters into a single dict.
    All values come from reference_outputs (ground truth) or outputs (predictions).

    Returns:
        {
            "task_id": str,
            "task": str,
            "solution": str,
            "retrieval_context": str,
            "final_code": str,
            "solution_run": Dict[str, Any],
            "execution_output": str,
            "correct_statements": List[str],
        }
    """
    task_id = (
        str(outputs.get("task_id") or "").strip()
        or str(reference_outputs.get("task_id") or "").strip()
        or str(inputs.get("task_id") or "").strip()
    )

    retrieval_context = str(outputs.get("retrieval_context") or "")
    final_code = str(outputs.get("final_code") or outputs.get("generated_code") or "")
    execution_output = str(outputs.get("execution_output") or "")

    solution_run = outputs.get("execution") or outputs.get("solution_run") or {}
    if not isinstance(solution_run, dict):
        solution_run = {}

    task = reference_outputs.get("task", "")
    solution = reference_outputs.get("solution", "")
    statements = reference_outputs.get("correct_statements") or []
    if not isinstance(statements, list):
        statements = []

    return {
        "task_id": task_id,
        "task": task,
        "solution": solution,
        "retrieval_context": retrieval_context,
        "final_code": final_code,
        "solution_run": solution_run,
        "execution_output": execution_output,
        "correct_statements": statements,
    }


__all__ = [
    "EvaluationSettings",
    "StandardEvaluationSettings",
    "_extract_execution_error",
    "_get_value",
    "build_node_eval_state",
    "_synthesize_eval_context",
    "build_reference_outputs",
    "ensure_dataset",
    "extract_task_prompt",
    "load_eval_tasks_or_raise",
]
