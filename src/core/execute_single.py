from dataclasses import asdict
from typing import Any, Dict, Optional, cast
import warnings

from langchain_core.runnables import RunnableConfig

from ..models.config import AppConfig
from ..models.schema import PipelineState, generate_default_state
from ..utils.execution_utils import generate_thread_id


def _to_metadata_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return cast(Dict[str, Any], value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return cast(Dict[str, Any], dict(value.__dict__))
    warnings.warn(f"Unable to convert {value} to dict for metadata; using empty dict instead.")
    return {}


def _build_running_config(
    *,
    task_id: str,
    thread_id: str,
    app_config: AppConfig,
    qdrant_client: Any,
) -> RunnableConfig:
    return RunnableConfig(
        configurable={
            "thread_id": thread_id,
            "pipeline_params": app_config.pipeline,
            "static_params": app_config.static,
            "agent_params": app_config.agent,
            "rag_build_config": app_config.rag,
            "qdrant_client": qdrant_client,
        },
        metadata={
            "task_id": task_id,
            "pipeline_params": _to_metadata_dict(app_config.pipeline),
            "static_params": _to_metadata_dict(app_config.static),
            "agent_params": _to_metadata_dict(app_config.agent),
            "rag_build_config": asdict(app_config.rag),
        },
    )


def _extract_prompt(task_data: Dict[str, Any]) -> str:
    return cast(
        str,
        task_data.get("query")
        or task_data.get("user_prompt")
        or task_data.get("task")
        or "",
    )


async def _execute_single_task(
    *,
    task_id: str,
    task_data: Dict[str, Any],
    app_config: AppConfig,
    pipeline: Any,
    thread_id: str,
    qdrant_client: Any = None,
) -> Dict[str, Any]:
    prompt = _extract_prompt(task_data)
    initial_state = generate_default_state(task_id=task_id, user_prompt=prompt)

    try:
        final_state = await pipeline.ainvoke(
            initial_state,
            config=_build_running_config(
                task_id=task_id,
                thread_id=thread_id,
                app_config=app_config,
                qdrant_client=qdrant_client,
            ),
        )
    except Exception as exc:
        return {**initial_state, "error": str(exc)}

    final_state = cast(PipelineState, final_state)
    execution = final_state.get("solution_run") or {}
    passed = bool(execution.get("passed", False)) if isinstance(execution, dict) else False
    print(f"{task_id}: {'PASS' if passed else 'FAIL'}")
    return cast(Dict[str, Any], final_state)


async def execute_single(
    *,
    tasks: Dict[str, Dict[str, Any]],
    app_config: AppConfig,
    pipeline: Any,
    thread_id: Optional[str] = None,
    qdrant_client: Any = None,
) -> Dict[str, Dict[str, Any]]:
    if len(tasks) != 1:
        raise ValueError("execute_single expects exactly one task in the tasks dictionary.")

    task_id, task_data = next(iter(tasks.items()))
    resolved_thread_id = thread_id or generate_thread_id(task_id)
    single_result = await _execute_single_task(
        task_id=task_id,
        task_data=task_data,
        app_config=app_config,
        pipeline=pipeline,
        thread_id=resolved_thread_id,
        qdrant_client=qdrant_client,
    )
    return {task_id: single_result}
