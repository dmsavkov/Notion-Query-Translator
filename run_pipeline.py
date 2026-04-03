import asyncio
from dataclasses import asdict
import warnings
from typing import Any, Dict, Optional, cast
import datetime

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig

from src.nodes import (
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
from src.all_functionality import load_eval_tasks
from src.routing import (
    route_after_codegen,
    route_after_execute,
    route_after_precheck,
    route_after_reflect,
)
from src.schema import (
    AgentParams,
    CliParams,
    PipelineParams,
    PipelineState,
    RagBuildConfig,
    StaticParams,
    build_cli_eval_tasks,
    generate_default_state,
)
from evals.test_dbs_script import provision_infrastructure


def generate_thread_id(prefix: Optional[str] = None) -> str:
    right_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = f"{prefix}_{right_now}" if prefix else right_now
    return unique_id


def build_pipeline():
    graph = StateGraph(PipelineState)
    # All nodes receive (state, config) parameters from LangGraph
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


async def run(
    eval_tasks: Dict[str, Dict[str, Any]],
    static_params: StaticParams,
    pipeline_params: PipelineParams,
    agent_params: AgentParams,
    rag_build_config: RagBuildConfig,
) -> Dict[str, Dict[str, Any]]:
    try:
        async with AsyncSqliteSaver.from_conn_string(static_params.sqlite_saver_path) as checkpointer:
            pipeline = build_pipeline().compile(checkpointer=checkpointer)
            semaphore = asyncio.Semaphore(static_params.max_concurrency)

            async def _run_task(task_id: str, task_data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
                async with semaphore:
                    try:
                        final_state = await execute_single_run(
                            task_id=task_id,
                            task_data=task_data,
                            static_params=static_params,
                            pipeline_params=pipeline_params,
                            agent_params=agent_params,
                            rag_build_config=rag_build_config,
                            pipeline=pipeline,
                        )
                        # State is now self-contained
                        execution_output = final_state.get("execution_output", "")
                        final_code = str(final_state.get("final_code") or final_state.get("generated_code") or "")
                        passed = final_state.get("solution_run", {}).get("passed", False)
                        
                        print(f"{task_id}: {'PASS' if passed else 'FAIL'}")
                        return task_id, {
                            "passed": passed,
                            "final_code": final_code,
                            "trials": final_state.get("trials", []),
                            "output": execution_output,
                            "execution": final_state.get("solution_run", {}),
                            "function_name": str(final_state.get("function_name", "")),
                        }
                    except Exception as e:
                        print(f"{task_id}: Exception occurred")
                        print(e)
                        return task_id, {
                            "passed": False,
                            "final_code": "",
                            "trials": [],
                            "error": str(e),
                        } 

            task_results = await asyncio.gather(
                *(_run_task(task_id, task_data) for task_id, task_data in eval_tasks.items())
            )

            return {task_id: result for task_id, result in task_results}
    finally:
        from src.rag_utils import close_qdrant_client_safely
        close_qdrant_client_safely()


async def main(
    static_params: Optional[StaticParams] = None,
    pipeline_params: Optional[PipelineParams] = None,
    agent_params: Optional[AgentParams] = None,
    rag_build_config: Optional[RagBuildConfig] = None,
    cli_params: Optional[CliParams] = None,
    dev_mode: bool = True,
) -> Dict[str, Dict[str, Any]]:
    final_static_params = static_params or StaticParams()
    final_pipeline_params = pipeline_params or PipelineParams()
    final_agent_params = agent_params or AgentParams()
    final_rag_build_config = rag_build_config or RagBuildConfig()
    final_eval_tasks = None

    if cli_params is not None:
        final_pipeline_params = final_pipeline_params.model_copy(
            update={"minimal": not cli_params.think}
        )

    if cli_params is not None and dev_mode:
        warnings.warn(
            "cli_params were provided while dev_mode=True; development mode is turned off"
            "and CLI task input is used.",
            stacklevel=2,
        )
        
    if cli_params is not None:
        final_eval_tasks = build_cli_eval_tasks(cli_params)
    
    elif dev_mode:
        print("Setting up test infrastructure...")
        provision_infrastructure()
        print("Test infrastructure ready.\n")
        
        final_eval_tasks = load_eval_tasks(
            evals_dir=final_static_params.evals_dir,
            case_type=final_static_params.case_type,
        )
            
    if final_eval_tasks is None:
        raise ValueError("No evaluation tasks were provided or loaded.")

    return await run(
        eval_tasks=final_eval_tasks,
        static_params=final_static_params,
        pipeline_params=final_pipeline_params,
        agent_params=final_agent_params,
        rag_build_config=final_rag_build_config,
    )


if __name__ == "__main__":
    asyncio.run(main())
