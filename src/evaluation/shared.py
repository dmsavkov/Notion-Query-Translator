import asyncio
from contextlib import AsyncExitStack, suppress
import json
import warnings
from typing import Any, Callable, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langsmith import Client
from langsmith.evaluation import aevaluate

from ..core.lifecycle import build_pipeline, run_with_lifecycle
from ..core.execute_single import _build_running_config
from ..models.config import AppConfig
from ..models.schema import AgentParams, PipelineParams, RagBuildConfig, StaticParams, generate_default_state
from ..nodes import cleanup_sandbox_node
from ..utils.openai_utils import openai_client_session
from .utils import StandardEvaluationSettings, _extract_execution_error, ensure_dataset, load_eval_tasks_or_raise


class ExactMatchEvaluator:
    """Exact-match evaluator for selected output keys against reference outputs."""

    def __init__(
        self,
        keys_to_check: List[str],
        *,
        metric_key: str = "exact_match_score",
        output_container_key: Optional[str] = None,
    ):
        self.keys_to_check = list(keys_to_check)
        self.metric_key = metric_key
        self.output_container_key = output_container_key

    def _resolve_output_source(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.output_container_key:
            return outputs

        container = outputs.get(self.output_container_key)
        if isinstance(container, dict):
            return container
        return {}

    async def __call__(
        self, *, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        expected = reference_outputs if isinstance(reference_outputs, dict) else {}
        predicted = outputs if isinstance(outputs, dict) else {}
        predicted_source = self._resolve_output_source(predicted)

        checked = 0
        matched = 0
        details: List[Dict[str, Any]] = []

        for key in self.keys_to_check:
            if key not in expected:
                details.append({"key": key, "expected": None, "predicted": predicted_source.get(key), "match": None, "checked": False})
                continue

            checked += 1
            expected_value = expected.get(key)
            predicted_value = predicted_source.get(key)
            is_match = predicted_value == expected_value
            if is_match:
                matched += 1

            details.append(
                {
                    "key": key,
                    "expected": expected_value,
                    "predicted": predicted_value,
                    "match": is_match,
                    "checked": True,
                }
            )

        score = 1.0 if checked == 0 else matched / float(checked)
        task_id = str(
            predicted.get("task_id")
            or expected.get("task_id")
            or inputs.get("task_id")
            or ""
        ).strip()

        return {
            "key": self.metric_key,
            "score": score,
            "comment": json.dumps(
                {
                    "task_id": task_id,
                    "checked": checked,
                    "matched": matched,
                    "details": details,
                }
            ),
        }


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

    post_dataset_sync_delay_seconds = float(getattr(settings, "post_dataset_sync_delay_seconds", 0.0) or 0.0)
    if post_dataset_sync_delay_seconds > 0:
        await asyncio.sleep(post_dataset_sync_delay_seconds)

    print(f"\n[Executing Eval] Dataset: {settings.dataset_name} | Prefix: {settings.experiment_prefix}")

    failed_executions: List[Dict[str, str]] = []
    async with openai_client_session():
        results_iterator = await aevaluate(
            target,
            data=settings.dataset_name,
            evaluators=evaluators,
            experiment_prefix=settings.experiment_prefix,
            max_concurrency=settings.eval_max_concurrency,
        )

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
        # Keep flattened top-level input keys as well without overriding nested state keys.
        for key, value in inputs.items():
            if key not in {"input_state", "reference_outputs", "task_id"} and key not in task_payload:
                task_payload[key] = value
        task_id = str(inputs.get("task_id") or task_payload.get("task_id") or "").strip()
        task_query = str(task_payload.get("query") or task_payload.get("user_prompt") or task_payload.get("task") or "").strip()
        return task_id, task_payload, task_query

    task_payload = {
        key: value
        for key, value in inputs.items()
        if key not in {"reference_outputs", "task_id"}
    }
    task_id = str(inputs.get("task_id") or task_payload.get("task_id") or "").strip()
    task_query = str(task_payload.get("query") or task_payload.get("user_prompt") or task_payload.get("task") or "").strip()
    return task_id, task_payload, task_query


def _seed_pipeline_state(*, task_id: str, task_payload: Dict[str, Any], task_query: str) -> Dict[str, Any]:
    state = dict(generate_default_state())
    for key in state.keys():
        if key in task_payload:
            state[key] = task_payload[key]

    if not str(state.get("user_prompt") or "").strip():
        state["user_prompt"] = task_query
    state["task_id"] = task_id
    return state


def _normalize_interrupt_nodes(value: Any, fallback: Optional[List[str]]) -> List[str]:
    if isinstance(value, str):
        normalized = [value.strip()] if value.strip() else []
        return normalized or list(fallback or [])

    if isinstance(value, (list, tuple, set)):
        normalized = [str(item).strip() for item in value if str(item).strip()]
        return normalized or list(fallback or [])

    return list(fallback or [])


def make_partial_live_eval_target(
    static_params: StaticParams,
    pipeline_params: PipelineParams,
    *,
    agent_params: Optional[AgentParams] = None,
    rag_build_config: Optional[RagBuildConfig] = None,
    start_as_node: str = "plan",
    interrupt_before: Optional[List[str]] = None,
):
    """
    Build a LangSmith target that runs a partial pipeline trajectory.

    For each dataset sample:
    1) Build a state seed from inputs
    2) Checkpoint that seed as if `start_as_node` just completed
    3) Resume graph execution until END or interrupt_before trigger
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

        thread_id = f"{task_id}_partial_live"
        if not task_query:
            return _error_target_output(
                task_id=task_id,
                thread_id=thread_id,
                error=f"No prompt found for task_id='{task_id}' in dataset inputs or task specs.",
            )

        resolved_start_node = str(task_payload.get("start_as_node") or start_as_node).strip() or start_as_node
        resolved_interrupts = _normalize_interrupt_nodes(task_payload.get("interrupt_before"), interrupt_before)

        initial_state = _seed_pipeline_state(task_id=task_id, task_payload=task_payload, task_query=task_query)

        try:
            async with AsyncExitStack() as stack:
                checkpointer = await stack.enter_async_context(
                    AsyncSqliteSaver.from_conn_string(app_config.static.sqlite_saver_path)
                )

                qdrant_client = None
                if app_config.static.context_used == "dynamic" and app_config.static.max_concurrency > 1:
                    from ..utils.rag_utils import qdrant_client_context

                    qdrant_client = await stack.enter_async_context(
                        qdrant_client_context(app_config.rag.qdrant_path)
                    )

                pipeline = build_pipeline().compile(
                    checkpointer=checkpointer,
                    interrupt_before=resolved_interrupts,
                )

                config = _build_running_config(
                    task_id=task_id,
                    thread_id=thread_id,
                    app_config=app_config,
                    qdrant_client=qdrant_client,
                    page_cache=None,
                )

                await pipeline.aupdate_state(config, initial_state, as_node=resolved_start_node)
                await pipeline.ainvoke(None, config=config)
                snapshot = await pipeline.aget_state(config)
                final_state = dict(snapshot.values)

                with suppress(Exception):
                    if final_state.get("sandbox_id"):
                        cleanup_update = await cleanup_sandbox_node(final_state, config)
                        if isinstance(cleanup_update, dict):
                            final_state.update(cleanup_update)
        except Exception as exc:
            warnings.warn(
                f"Partial live target execution failed for task_id='{task_id}': {exc}",
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
            "meta": final_state.get("meta") or {},
            "security": final_state.get("security") or {},
            "execution": execution,
            "solution_run": execution,
            "execution_output": str(final_state.get("execution_output") or ""),
            "function_name": str(final_state.get("function_name") or ""),
            "terminal_status": str(final_state.get("terminal_status") or ""),
            "trial_num": int(final_state.get("trial_num", 0) or 0),
            "trials": final_state.get("trials") or [],
            "error": "",
        }

    return target


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
            "meta": final_state.get("meta") or {},
            "security": final_state.get("security") or {},
            "execution": execution,
            "solution_run": execution,
            "execution_output": str(final_state.get("execution_output") or ""),
            "function_name": str(final_state.get("function_name") or ""),
            "trials": final_state.get("trials") or [],
            "error": "",
        }

    return target


__all__ = [
    "ExactMatchEvaluator",
    "evaluation_orchestration",
    "make_live_eval_target",
    "make_partial_live_eval_target",
]
