import datetime
from dataclasses import asdict
from typing import Any, Dict, Optional, cast

from langchain_core.runnables import RunnableConfig

from .schema import (
    AgentParams,
    PipelineParams,
    PipelineState,
    RagBuildConfig,
    StaticParams,
    generate_default_state,
)


def generate_thread_id(prefix: Optional[str] = None) -> str:
    right_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = f"{prefix}_{right_now}" if prefix else right_now
    return unique_id


async def execute_single_run(
    *,
    task_id: str,
    task_data: Dict[str, Any],
    static_params: StaticParams,
    pipeline_params: PipelineParams,
    agent_params: AgentParams,
    rag_build_config: RagBuildConfig,
    thread_id: Optional[str] = None,
    pipeline: Optional[Any] = None,
    checkpointer: Optional[Any] = None,
) -> PipelineState:
    prompt = (
        task_data.get("query")
        or task_data.get("user_prompt")
        or task_data.get("task")
        or ""
    )
    initial_state = generate_default_state(task_id=task_id, user_prompt=prompt)

    configurable = {
        "thread_id": thread_id or generate_thread_id(prefix=task_id),
        "pipeline_params": pipeline_params.model_dump(),
        "static_params": static_params.model_dump(),
        "agent_params": agent_params.model_dump(),
        "build_rag": asdict(rag_build_config),
    }

    runnable_pipeline = pipeline
    if runnable_pipeline is None:
        from run_pipeline import build_pipeline

        runnable_pipeline = build_pipeline().compile(checkpointer=checkpointer)

    final_state = await runnable_pipeline.ainvoke(
        initial_state,
        config=RunnableConfig(
            configurable=configurable,
            metadata=configurable,
        ),
    )
    return cast(PipelineState, final_state)
