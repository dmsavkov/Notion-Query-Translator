import asyncio
import json
import warnings
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client
from langsmith.evaluation import aevaluate
from pydantic import BaseModel, ConfigDict, computed_field

from src.running_utils import execute_single_run
from src.all_functionality import load_eval_tasks
from src.error_analysis import HumanConfig, main as run_error_analysis_main
from src.evaluator import Evaluator
from src.schema import AgentParams, PipelineParams, RagBuildConfig, StaticParams

class EvaluationSettings(BaseModel):
    experiment_prefix: str = "COMPLEX CONTEXT UPDATED: personal comprehensive + top25_20220628 + scratch, refl3."
    evals_case_type: str = "complex"
    judge_model_name: str = "gemini-3.1-flash-lite-preview"
    eval_max_concurrency: int = 5
    run_error_analysis_after_eval: bool = True
    evals_dir: str = "evals"

    model_config = ConfigDict(frozen=True)

    @computed_field
    @property
    def dataset_name(self) -> str:
        if self.evals_case_type == "complex":
            return "Dataset v4."
        if self.evals_case_type == "simple":
            return "Dataset v1."
        if self.evals_case_type == "all":
            return "Dataset v3."
        raise ValueError(f"Unsupported evals_case_type: {self.evals_case_type}")

    def build_client(self) -> Client:
        return Client()

    def build_core_evaluator(self) -> Evaluator:
        return Evaluator(default_judge_model=self.judge_model_name)

    def load_eval_tasks(self) -> Dict[str, Dict[str, Any]]:
        tasks = load_eval_tasks(self.evals_dir, case_type=cast(Any, self.evals_case_type))
        if not tasks:
            raise ValueError(
                f"No evaluation tasks loaded for case_type='{self.evals_case_type}'. "
                "Check evals directory and case filters."
            )
        return tasks


SETTINGS = EvaluationSettings()


def _extract_task_prompt(task_spec: Dict[str, Any]) -> str:
    return str(task_spec.get("query") or task_spec.get("user_prompt") or task_spec.get("task") or "").strip()


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

    solution_run = outputs.get("execution") or {}
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


def _build_reference_outputs(task_id: str, task_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Reference outputs attached to dataset examples (ground truth only)."""
    task = _extract_task_prompt(task_spec)
    return {
        "task_id": task_id,
        "task": task,
        "solution": task_spec.get("solution", ""),
        "correct_statements": task_spec.get("correct_statements", []) or [],
    }


def _ensure_dataset(client: Client, dataset_name: str, task_specs: Dict[str, Dict[str, Any]]) -> Any:
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
                "query": _extract_task_prompt(task_spec),
            },
            outputs=_build_reference_outputs(task_id, task_spec),
            dataset_id=dataset.id,
        )
        created += 1

    print(f"Dataset '{dataset_name}' ready. Existing: {len(existing_by_task_id)} | Created now: {created}")
    return dataset


def _error_target_output(task_id: str, thread_id: str, error: str) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "thread_id": thread_id,
        "retrieval_context": "",
        "final_code": "",
        "execution": {},
        "execution_output": "",
        "function_name": "",
        "error": error,
    }


def make_live_eval_target(
    static_params: StaticParams,
    pipeline_params: PipelineParams,
    *,
    task_specs: Dict[str, Dict[str, Any]],
    agent_params: Optional[AgentParams] = None,
    rag_build_config: Optional[RagBuildConfig] = None,
):
    """
    Build a LangSmith target that executes one live pipeline run per dataset sample.

    The target uses a fresh in-memory checkpointer per invocation and returns
    only evaluator-required output fields.
    """
    final_agent_params = agent_params or AgentParams()
    final_rag_build_config = rag_build_config or RagBuildConfig()

    async def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        task_id = str(inputs.get("task_id") or "").strip()
        if not task_id:
            return _error_target_output(task_id="", thread_id="", error="Missing task_id in dataset inputs.")

        task_query = str(inputs.get("query") or inputs.get("user_prompt") or inputs.get("task") or "").strip()
        if not task_query:
            task_query = _extract_task_prompt(task_specs.get(task_id, {}))

        thread_id = f"{task_id}_{uuid4().hex}"
        if not task_query:
            return _error_target_output(
                task_id=task_id,
                thread_id=thread_id,
                error=f"No prompt found for task_id='{task_id}' in dataset inputs or task specs.",
            )

        try:
            final_state = await execute_single_run(
                task_id=task_id,
                task_data={"query": task_query},
                static_params=static_params,
                pipeline_params=pipeline_params,
                agent_params=final_agent_params,
                rag_build_config=final_rag_build_config,
                thread_id=thread_id,
                checkpointer=MemorySaver(),
            )
        except Exception as exc:
            warnings.warn(
                f"Live target execution failed for task_id='{task_id}': {exc}",
                stacklevel=2,
            )
            return _error_target_output(task_id=task_id, thread_id=thread_id, error=str(exc))

        execution = final_state.get("solution_run") or {}
        if not isinstance(execution, dict):
            execution = {}

        return {
            "task_id": task_id,
            "thread_id": thread_id,
            "retrieval_context": str(final_state.get("retrieval_context") or ""),
            "final_code": str(final_state.get("final_code") or final_state.get("generated_code") or ""),
            "execution": execution,
            "execution_output": str(final_state.get("execution_output") or ""),
            "function_name": str(final_state.get("function_name") or ""),
            "error": "",
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

        if not statements:
            warnings.warn(
                f"Task '{task_id}' has empty correct_statements for rag_statements_score.",
                stacklevel=2,
            )

        status_items = await self.evaluator.eval_context_statements(
            context=retrieval_context,
            statements=statements,
            judge_model_name=self.judge_model_name,
        )

        if not status_items:
            warnings.warn(
                f"rag_statements_score evaluator returned empty output for task '{task_id}'. Scoring as 0.0.",
                stacklevel=2,
            )
            return {
                "key": "rag_statements_score",
                "score": 0.0,
                "comment": json.dumps({"error": "Empty evaluator output", "task_id": task_id}),
            }

        score, _ = _present_score(status_items)

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

        if not statements:
            warnings.warn(
                f"Task '{task_id}' has empty correct_statements for code_statements_score.",
                stacklevel=2,
            )

        status_items = await self.evaluator.eval_context_statements(
            context=final_code,
            statements=statements,
            judge_model_name=self.judge_model_name,
        )

        if not status_items:
            warnings.warn(
                f"code_statements_score evaluator returned empty output for task '{task_id}'. Scoring as 0.0.",
                stacklevel=2,
            )
            return {
                "key": "code_statements_score",
                "score": 0.0,
                "comment": json.dumps({"error": "Empty evaluator output", "task_id": task_id}),
            }

        score, _ = _present_score(status_items)

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

        result = await self.evaluator.eval_code_exec(
            execution=cast(Dict[str, Any], ctx.get("solution_run") or {}),
            execution_output=str(ctx.get("execution_output") or ""),
        )

        if not result:
            warnings.warn(
                f"code_execution_score evaluator returned empty output for task '{task_id}'. Scoring as 0.0.",
                stacklevel=2,
            )
            return {
                "key": "code_execution_score",
                "score": 0.0,
                "comment": json.dumps({"error": "Empty evaluator output", "task_id": task_id}),
            }

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


def _build_evaluators(core_evaluator: Evaluator, settings: EvaluationSettings) -> List[Any]:
    rag_eval = RagStatementsEvaluator(core_evaluator, judge_model_name=settings.judge_model_name)
    code_statements_eval = CodeStatementsEvaluator(core_evaluator, judge_model_name=settings.judge_model_name)
    code_execution_eval = CodeExecutionEvaluator(core_evaluator)
    return [
        rag_eval.__call__,
        code_statements_eval.__call__,
        code_execution_eval.__call__,
    ]


async def main(settings: Optional[EvaluationSettings] = None) -> None:
    final_settings = settings or SETTINGS
    static_params = StaticParams(case_type=cast(Any, final_settings.evals_case_type))
    pipeline_params = PipelineParams(minimal=False)
    agent_params = AgentParams()
    rag_build_config = RagBuildConfig()
    client = final_settings.build_client()
    core_evaluator = final_settings.build_core_evaluator()
    eval_tasks = final_settings.load_eval_tasks()

    _ensure_dataset(client, final_settings.dataset_name, eval_tasks)

    target = make_live_eval_target(
        static_params,
        pipeline_params,
        task_specs=eval_tasks,
        agent_params=agent_params,
        rag_build_config=rag_build_config,
    )

    evaluators: List[Any] = _build_evaluators(core_evaluator, final_settings)

    results = await aevaluate(
        target,
        data=final_settings.dataset_name,
        evaluators=cast(Any, evaluators),
        experiment_prefix=final_settings.experiment_prefix,
        max_concurrency=final_settings.eval_max_concurrency,
    )

    print(results)

    if final_settings.run_error_analysis_after_eval:
        try:
            error_analysis_result = await asyncio.to_thread(
                run_error_analysis_main,
                final_settings.experiment_prefix,
                HumanConfig(),
                final_settings.dataset_name,
            )
            print(
                "Error analysis complete: "
                f"page_title={error_analysis_result.get('page_title', '')}, "
                f"page_id={error_analysis_result.get('page_id', '')}, "
                f"record_count={error_analysis_result.get('record_count', 0)}"
            )
            print("Second-prompt payload copied to clipboard.")
        except Exception as exc:
            print(f"Post-eval error analysis failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())





