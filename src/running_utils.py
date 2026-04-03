import datetime
from dataclasses import asdict
from typing import Any, Dict, Optional, cast

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig

from .nodes import (
    codegen_node,
    execute_node,
    precheck_general_node,
    precheck_join_node,
    precheck_security_node,
    malovolent_request_node,
    plan_node,
    reflect_node,
    retrieve_node,
)
from .routing import (
    route_after_codegen,
    route_after_execute,
    route_after_precheck,
    route_after_reflect,
)
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


def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)
    # All nodes receive (state, config) parameters from LangGraph.
    graph.add_node("precheck_general", cast(Any, precheck_general_node))
    graph.add_node("precheck_security", cast(Any, precheck_security_node))
    graph.add_node("precheck_join", cast(Any, precheck_join_node))
    graph.add_node("malovolent_request", cast(Any, malovolent_request_node))
    graph.add_node("retrieve", cast(Any, retrieve_node))
    graph.add_node("plan", cast(Any, plan_node))
    graph.add_node("codegen", cast(Any, codegen_node))
    graph.add_node("execute", cast(Any, execute_node))
    graph.add_node("reflect", cast(Any, reflect_node))

    graph.add_edge(START, "precheck_general")
    graph.add_edge(START, "precheck_security")
    graph.add_edge(["precheck_general", "precheck_security"], "precheck_join")
    graph.add_conditional_edges("precheck_join", route_after_precheck)
    graph.add_edge("malovolent_request", END)
    graph.add_edge("retrieve", "plan")
    graph.add_edge("plan", "codegen")
    graph.add_conditional_edges("codegen", route_after_codegen)
    graph.add_conditional_edges("execute", route_after_execute)
    graph.add_conditional_edges("reflect", route_after_reflect)

    return graph


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
        runnable_pipeline = build_pipeline().compile(checkpointer=checkpointer)

    final_state = await runnable_pipeline.ainvoke(
        initial_state,
        config=RunnableConfig(
            configurable=configurable,
            metadata=configurable,
        ),
    )
    return cast(PipelineState, final_state)
