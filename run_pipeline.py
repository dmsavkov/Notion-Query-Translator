import asyncio
import warnings
from typing import Any, Dict, Optional

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.all_functionality import load_eval_tasks
from src.routing import (
    route_after_codegen,
    route_after_execute,
    route_after_precheck,
    route_after_reflect,
)
from src.running_utils import build_pipeline, execute_single_run, generate_thread_id
from src.schema import (
    AgentParams,
    CliParams,
    PipelineParams,
    PipelineState,
    RagBuildConfig,
    StaticParams,
    build_cli_eval_tasks,
)
from evals.test_dbs_script import provision_infrastructure


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
