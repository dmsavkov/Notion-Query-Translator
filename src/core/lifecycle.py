from contextlib import AsyncExitStack
from typing import Any, Dict, cast

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from ..models.config import AppConfig
from ..models.schema import PipelineState
from ..utils.execution_utils import generate_thread_id
from ..nodes import (
    codegen_node,
    execute_node,
    malovolent_request_node,
    plan_node,
    precheck_general_node,
    precheck_join_node,
    precheck_security_node,
    reflect_node,
    retrieve_node,
)
from ..routing import (
    route_after_codegen,
    route_after_execute,
    route_after_precheck,
    route_after_reflect,
)
from .execute_batch import execute_batch
from .execute_single import execute_single


def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)
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


async def run_with_lifecycle(
    *,
    tasks: Dict[str, Dict[str, Any]],
    app_config: AppConfig,
) -> Dict[str, Dict[str, Any]]:
    if not tasks:
        raise ValueError("No tasks provided to run_with_lifecycle.")

    async with AsyncExitStack() as stack:
        checkpointer = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(app_config.static.sqlite_saver_path)
        )

        qdrant_client = None
        if (
            app_config.static.context_used == "dynamic"
            and app_config.static.max_concurrency > 1
        ):
            from ..utils.rag_utils import qdrant_client_context

            qdrant_client = await stack.enter_async_context(
                qdrant_client_context(app_config.rag.qdrant_path)
            )

        # Pipeline compilation is costly; compile once in lifecycle and pass down.
        pipeline = build_pipeline().compile(checkpointer=checkpointer)

        if len(tasks) > 1 and app_config.static.max_concurrency > 1:
            return await execute_batch(
                tasks=tasks,
                app_config=app_config,
                pipeline=pipeline,
                qdrant_client=qdrant_client,
            )

        if len(tasks) == 1:
            return await execute_single(
                tasks=tasks,
                app_config=app_config,
                pipeline=pipeline,
                qdrant_client=qdrant_client,
            )

        merged_results: Dict[str, Dict[str, Any]] = {}
        for task_id, task_data in tasks.items():
            task_result = await execute_single(
                tasks={task_id: task_data},
                app_config=app_config,
                pipeline=pipeline,
                qdrant_client=qdrant_client,
            )
            merged_results.update(task_result)

        return merged_results
