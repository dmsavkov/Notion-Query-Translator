import asyncio
import json
from typing import Any, Dict, List, Optional, cast

from langsmith import Client 
from langsmith.evaluation import aevaluate
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.runnables import RunnableConfig

from run_pipeline import StaticParams
from src.all_functionality import load_eval_tasks
from src.evaluator import Evaluator

# Global setup
EXPERIMENT_PREFIX = "COMPLEX CONTEXT UPDATED: personal comprehensive + top25_20220628."  # for grouping runs in LangSmith
EVALS_CASE_TYPE = "complex"
JUDGE_MODEL_NAME = "gemini-3.1-flash-lite-preview"
EVAL_MAX_CONCURRENCY = 5

DATASET_NAME = ""
N_LAST_RUNS = 0

if EVALS_CASE_TYPE == "complex":
    DATASET_NAME = "Dataset v4." # updated
    N_LAST_RUNS = 5
elif EVALS_CASE_TYPE == "simple": 
    DATASET_NAME = "Dataset v1."
    N_LAST_RUNS = 6
elif EVALS_CASE_TYPE == "all":
    DATASET_NAME = "Dataset v3."
    N_LAST_RUNS = 11
else:
    raise ValueError(f"Unsupported EVALS_CASE_TYPE: {EVALS_CASE_TYPE}")
    
THREAD_PREFIX_FILTERS: List[str] = []  # [] means no filtering


client = Client()
core_evaluator = Evaluator(default_judge_model=JUDGE_MODEL_NAME)
eval_tasks = load_eval_tasks("evals", case_type=EVALS_CASE_TYPE)

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
            "task": str (the task description),
            "solution": str,
            "retrieval_context": str,
            "final_code": str,
            "correct_statements": List[str],
        }
    """
    # task_id priority: target output > reference > inputs
    task_id = (
        str(outputs.get("task_id") or "").strip()
        or str(reference_outputs.get("task_id") or "").strip()
        or str(inputs.get("task_id") or "").strip()
    )
    
    state = outputs.get("pre_computed_state", {}) or {}
    retrieval_context = state.get("retrieval_context", "")
    final_code = state.get("final_code") or state.get("generated_code") or ""

    # Extract all ground truth from reference_outputs
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
        "correct_statements": statements,
    }


def _build_reference_outputs(task_id: str, task_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Reference outputs attached to dataset examples (ground truth only)."""
    task = task_spec.get("query") or task_spec.get("user_prompt") or task_spec.get("task") or ""
    return {
        "task_id": task_id,
        "task": task,
        "solution": task_spec.get("solution", ""),
        "correct_statements": task_spec.get("correct_statements", []) or [],
    }


def _ensure_dataset(dataset_name: str, task_specs: Dict[str, Dict[str, Any]]) -> Any:
    """
    Ensure a single stable dataset exists and contains one example per eval task.
    Existing examples are reused; missing ones are created.
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
            },
            outputs=_build_reference_outputs(task_id, task_spec),
            dataset_id=dataset.id,
        )
        created += 1

    print(f"Dataset '{dataset_name}' ready. Existing: {len(existing_by_task_id)} | Created now: {created}")
    return dataset


async def _load_recent_states(
    sqlite_saver_path: str,
    n_last_runs: int,
    thread_prefix_filters: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load the latest N unique thread states from checkpoint history.
    Returns dict mapping task_id -> state for direct lookup.
    """
    states: Dict[str, Dict[str, Any]] = {}
    if n_last_runs <= 0:
        return states

    filters = thread_prefix_filters or []

    async with AsyncSqliteSaver.from_conn_string(sqlite_saver_path) as checkpointer:
        thread_ids: List[str] = []
        seen_thread_ids = set()

        checkpoint_iter = checkpointer.alist(None)
        try:
            async for checkpoint in checkpoint_iter:
                thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
                if not thread_id or thread_id in seen_thread_ids:
                    continue

                if filters and not any(thread_id.startswith(prefix + "_") for prefix in filters):
                    continue

                seen_thread_ids.add(thread_id)
                thread_ids.append(thread_id)

                if len(thread_ids) >= n_last_runs:
                    break
        finally:
            # Explicitly close iterator when stopping early to avoid noisy GeneratorExit warnings.
            await checkpoint_iter.aclose()

        for thread_id in thread_ids:
            config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
            checkpoint_tuple = await checkpointer.aget_tuple(config)
            if not checkpoint_tuple:
                continue

            channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
            # Extract task_id from state (it's now part of the state)
            state_task_id = channel_values.get("task_id", "")
            
            if state_task_id and state_task_id not in states:
                states[state_task_id] = {
                    "thread_id": thread_id,
                    "task_id": state_task_id,
                    "pre_computed_state": channel_values,
                }

    return states


def _select_state_for_task(states: Dict[str, Dict[str, Any]], task_id: str) -> Optional[Dict[str, Any]]:
    """
    Direct lookup of state by task_id. Returns None if not found.
    """
    return states.get(task_id)


def make_precomputed_target(states: Dict[str, Dict[str, Any]]):
    """
    Build a target that maps dataset task_id to previously checkpointed states.
    """

    def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        task_id = str(inputs.get("task_id") or "").strip()
        selected_state = _select_state_for_task(states, task_id)

        if selected_state is None:
            return {
                "task_id": task_id,
                "thread_id": "",
                "pre_computed_state": {},
                "error": f"No matching checkpoint state found for task_id={task_id}",
            }

        return {
            "task_id": task_id,
            "thread_id": selected_state.get("thread_id", ""),
            "pre_computed_state": selected_state.get("pre_computed_state", {}),
        }

    return target


def _present_score(status_items: List[Dict[str, Any]]) -> tuple[float, int]:
    present_count = sum(
        1
        for item in status_items
        if str(item.get("status", "")).strip().lower() == "present"
    )
    score = (present_count / len(status_items)) if status_items else 0.0
    return score, present_count


class RagStatementsEvaluator:
    """Independent evaluator for statement presence in RAG retrieval context."""

    def __init__(self, evaluator: Evaluator, judge_model_name: str | None = None):
        self.evaluator = evaluator
        self.judge_model_name = judge_model_name

    async def _compute(
        self, *, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        ctx = _synthesize_eval_context(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
        )
        task_id = ctx["task_id"]
        retrieval_context = ctx["retrieval_context"]
        statements = ctx["correct_statements"]

        status_items = await self.evaluator.eval_context_statements(
            context=retrieval_context,
            statements=statements,
            judge_model_name=self.judge_model_name,
        )

        score, present_count = _present_score(status_items)

        return {
            "key": "rag_statements_score",
            "score": score,
            "comment": json.dumps(status_items),
        }

    async def __call__(
        self, *, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return await self._compute(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)


class CodeStatementsEvaluator:
    """Independent evaluator for statement presence in generated code."""

    def __init__(self, evaluator: Evaluator, judge_model_name: str | None = None):
        self.evaluator = evaluator
        self.judge_model_name = judge_model_name

    async def _compute(
        self, *, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        ctx = _synthesize_eval_context(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
        )
        task_id = ctx["task_id"]
        final_code = ctx["final_code"]
        statements = ctx["correct_statements"]

        status_items = await self.evaluator.eval_context_statements(
            context=final_code,
            statements=statements,
            judge_model_name=self.judge_model_name,
        )

        score, present_count = _present_score(status_items)

        return {
            "key": "code_statements_score",
            "score": score,
            "comment": json.dumps(status_items),
        }

    async def __call__(
        self, *, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return await self._compute(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)


class CodeExecutionEvaluator:
    """Independent evaluator for generated code execution pass/fail."""

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

    async def _compute(
        self, *, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        ctx = _synthesize_eval_context(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
        )
        task_id = ctx["task_id"]
        final_code = ctx["final_code"]

        result = await self.evaluator.eval_code_exec(final_code)
        execution_pass = bool(result.get("pass", False))

        return {
            "key": "code_execution_score",
            "score": 1.0 if execution_pass else 0.0,
            "comment": json.dumps(result),
        }

    async def __call__(
        self, *, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return await self._compute(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)


rag_eval = RagStatementsEvaluator(core_evaluator, judge_model_name=JUDGE_MODEL_NAME)
code_statements_eval = CodeStatementsEvaluator(core_evaluator, judge_model_name=JUDGE_MODEL_NAME)
code_execution_eval = CodeExecutionEvaluator(core_evaluator)

async def _debug_one_state(states_to_evaluate: Dict[str, Dict[str, Any]]) -> None:
    if len(states_to_evaluate) < 1:
        print("No checkpoint states available")
        return

    # Pick first state in dict
    first_task_id = next(iter(states_to_evaluate))
    st = states_to_evaluate[first_task_id]
    res = await core_evaluator.eval_context_statements(
        context=st["pre_computed_state"].get("final_code", ""),
        statements=eval_tasks.get(first_task_id, {}).get("correct_statements", []),
    )
    print(f"Debug eval for task={first_task_id}:")
    print(res)


# asyncio.run(_debug_one_state(states_to_evaluate))

async def main():
    static_params = StaticParams()

    _ensure_dataset(DATASET_NAME, eval_tasks)

    states_to_evaluate = await _load_recent_states(
        sqlite_saver_path=static_params.sqlite_saver_path,
        n_last_runs=N_LAST_RUNS,
        thread_prefix_filters=THREAD_PREFIX_FILTERS,
    )

    print(f"Loaded checkpoint states: {len(states_to_evaluate)}/{N_LAST_RUNS} unique tasks")
    for task_id, state in sorted(states_to_evaluate.items()):
        print(f"  {task_id}: thread={state.get('thread_id', '')}")

    print("\nDataset -> Checkpoint mapping:")
    for task_id in sorted(eval_tasks.keys()):
        selected_state = _select_state_for_task(states_to_evaluate, task_id)
        selected_thread = selected_state.get("thread_id", "") if selected_state else "<not found>"
        status = "✓" if selected_state else "✗"
        print(f"  {status} task={task_id} -> thread={selected_thread}")

    target = make_precomputed_target(states_to_evaluate)

    async def async_target_wrapper(inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Run synchronous target in a worker thread so aevaluate can await it.
        return await asyncio.to_thread(target, inputs)

    evaluators: List[Any] = [
        rag_eval.__call__,
        code_statements_eval.__call__,
        code_execution_eval.__call__,
    ]

    results = await aevaluate(
        async_target_wrapper,
        data=DATASET_NAME,
        evaluators=cast(Any, evaluators),
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=EVAL_MAX_CONCURRENCY,
    )

    print(results)

if __name__ == "__main__":
    asyncio.run(main())





