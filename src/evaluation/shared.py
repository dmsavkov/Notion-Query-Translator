import asyncio
import warnings
from typing import Any, Callable, Dict, List, Optional

from langsmith import Client
from langsmith.evaluation import aevaluate

from ..core.lifecycle import run_with_lifecycle
from ..models.config import AppConfig
from ..models.schema import AgentParams, PipelineParams, RagBuildConfig, StaticParams
from .utils import StandardEvaluationSettings, _extract_execution_error, ensure_dataset, load_eval_tasks_or_raise


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

    print(f"\n[Executing Eval] Dataset: {settings.dataset_name} | Prefix: {settings.experiment_prefix}")

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


def _error_target_output(task_id: str, thread_id: str, error: str) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "thread_id": thread_id,
        "retrieval_context": "",
        "final_code": "",
        "execution": {},
        "solution_run": {},
        "execution_output": "",
        "function_name": "",
        "error": error,
    }


def _resolve_live_task_payload(inputs: Dict[str, Any]) -> tuple[str, Dict[str, Any], str]:
    input_state = inputs.get("input_state")
    if isinstance(input_state, dict):
        task_payload = dict(input_state)
        task_id = str(inputs.get("task_id") or task_payload.get("task_id") or "").strip()
        task_query = str(task_payload.get("query") or task_payload.get("user_prompt") or task_payload.get("task") or "").strip()
        return task_id, task_payload, task_query

    task_id = str(inputs.get("task_id") or "").strip()
    task_query = str(inputs.get("query") or inputs.get("user_prompt") or inputs.get("task") or "").strip()
    return task_id, {"query": task_query}, task_query


def make_live_eval_target(
    static_params: StaticParams,
    pipeline_params: PipelineParams,
    *,
    agent_params: Optional[AgentParams] = None,
    rag_build_config: Optional[RagBuildConfig] = None,
):
    """
    Build a LangSmith target that executes one live pipeline run per dataset sample.

    The target delegates lifecycle ownership to run_with_lifecycle, which creates
    the SQL checkpointer and any conditional shared resources.
    """
    final_agent_params = agent_params or AgentParams()
    final_rag_build_config = rag_build_config or RagBuildConfig()
    app_config = AppConfig(
        pipeline=pipeline_params,
        static=static_params,
        agent=final_agent_params,
        rag=final_rag_build_config,
    )

    async def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        task_id, task_payload, task_query = _resolve_live_task_payload(inputs)
        if not task_id:
            return _error_target_output(task_id="", thread_id="", error="Missing task_id in dataset inputs.")

        thread_id = f"{task_id}_live"
        if not task_query:
            return _error_target_output(
                task_id=task_id,
                thread_id=thread_id,
                error=f"No prompt found for task_id='{task_id}' in dataset inputs or task specs.",
            )

        try:
            result = await run_with_lifecycle(
                tasks={task_id: task_payload},
                app_config=app_config,
            )
            final_state = result[task_id]
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
            "solution_run": execution,
            "execution_output": str(final_state.get("execution_output") or ""),
            "function_name": str(final_state.get("function_name") or ""),
            "error": "",
        }

    return target


__all__ = [
    "evaluation_orchestration",
    "make_live_eval_target",
]
