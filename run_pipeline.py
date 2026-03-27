import asyncio
import contextlib
import io
import operator
from dataclasses import asdict
import warnings
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict, cast
import datetime

from pydantic import BaseModel, ConfigDict
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig

from build_rag import RagBuildConfig
from src.hardcoded_contexts import ContextUsed
from src.nodes import (
    codegen_node,
    execute_node,
    plan_node,
    reflect_node,
    retrieve_node,
)
from src.execution_utils import run_isolated_code
from src.all_functionality import load_eval_tasks
from src.rag_utils import close_qdrant_client_safely
from evals.test_dbs_script import provision_infrastructure


def generate_thread_id(prefix: Optional[str] = None) -> str:
    right_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = f"{prefix}_{right_now}" if prefix else right_now
    return unique_id


class CodeGeneratorParams(BaseModel):
    model_name: str = "gemini-3.1-flash-lite-preview"
    model_temperature: float = 0.3
    max_tokens: int = None

    model_config = ConfigDict(frozen=True)


class ReflectorParams(BaseModel):
    model_name: str = "gemma27"
    model_temperature: float = 0.2
    max_tokens: int = 1200

    model_config = ConfigDict(frozen=True)


class QueryTranslatorParams(BaseModel):
    model_name: str = "gemma4"
    model_temperature: float = 0.3
    max_tokens: int = None
    n_queries: int = 3
    top_k: int = 3
    top_k_total: int = 5
    query_method: Literal["multi_query", "cot_decompose", "domain_decompose"] = "cot_decompose"
    use_summarization: bool = True
    summarization_model_name: str = "gemma27"
    summarization_temperature: float = 0.2
    summarization_max_tokens: int = 200

    model_config = ConfigDict(frozen=True)


class RequestPlannerParams(BaseModel):
    model_name: str = "gemma27"
    model_temperature: float = 0.3
    max_tokens: int = None

    model_config = ConfigDict(frozen=True)


class AgentParams(BaseModel):
    """Per-node model and retrieval settings."""
    code_generator: CodeGeneratorParams = CodeGeneratorParams()
    reflector: ReflectorParams = ReflectorParams()
    query_translator: QueryTranslatorParams = QueryTranslatorParams()
    request_planner: RequestPlannerParams = RequestPlannerParams()

    model_config = ConfigDict(frozen=True)


class PipelineParams(BaseModel):
    """Dynamic parameters used during pipeline execution."""
    minimal: bool = False
    max_trials: int = 3

    model_config = ConfigDict(frozen=True)


class CliParams(BaseModel):
    """CLI-level parameters passed into the pipeline entrypoint."""
    user_prompt: str
    think: bool = False

    model_config = ConfigDict(frozen=True)


class StaticParams(BaseModel):
    """Static parameters for pipeline initialization."""
    evals_dir: str = "evals"
    output_dir: str = "evaluation_results"
    sqlite_saver_path: str = "data/checkpoints.sqlite"
    case_type: Literal["simple", "complex", "all"] = "complex"
    context_used: ContextUsed = "database_schema_report_comprehensive__notion_api_top25_20220628"
    enable_planning: bool = False
    max_concurrency: int = 6
    
    model_config = ConfigDict(frozen=True)


class PipelineState(TypedDict):
    task_id: str
    user_prompt: str
    retrieval_context: str
    request_plan: str
    general_info: str
    trial_num: int
    generated_code: str
    function_name: str
    solution_run: Dict[str, Any]
    execution_output: str
    reflection_context: List[str]
    feedback: str
    verdict: Dict[str, Any]
    trials: List[Dict[str, Any]]
    final_code: str
    queries: Annotated[List[str], operator.add]


def build_cli_eval_tasks(cli_params: CliParams) -> Dict[str, Dict[str, Any]]:
    return {
        "user_request": {
            "query": cli_params.user_prompt,
            "think": cli_params.think,
        }
    }


def generate_default_state(task_id: str, user_prompt: str) -> PipelineState:
    return {
        "task_id": task_id,
        "user_prompt": user_prompt,
        "retrieval_context": "",
        "request_plan": "",
        "general_info": "",
        "trial_num": 0,
        "generated_code": "",
        "function_name": "",
        "solution_run": {},
        "execution_output": "",
        "reflection_context": [],
        "feedback": "",
        "verdict": {},
        "trials": [],
        "final_code": "",
        "queries": [],
    }


def route_after_codegen(state: PipelineState, config: RunnableConfig) -> str:
    # Always execute at least once so pass/fail reflects real runtime behavior.
    return "execute"


def route_after_execute(state: PipelineState, config: RunnableConfig) -> str:
    cfg = cast(Dict[str, Any], config.get("configurable", {}))
    pipeline_params = cast(Dict[str, Any], cfg.get("pipeline_params", {}))
    minimal = bool(pipeline_params.get("minimal", False))
    if minimal:
        return END
    return "reflect"


def route_after_reflect(state: PipelineState, config: RunnableConfig) -> str:
    # Use the LLM's pass/fail verdict for routing
    if state.get("verdict", {}).get("pass", False):
        return END
    
    max_trials = config["configurable"]["pipeline_params"]["max_trials"]
    if state.get("trial_num", 0) >= max_trials:
        return END
    return "codegen"


def build_pipeline():
    graph = StateGraph(PipelineState)
    # All nodes receive (state, config) parameters from LangGraph
    graph.add_node("retrieve", cast(Any, retrieve_node))
    graph.add_node("plan", cast(Any, plan_node))
    graph.add_node("codegen", cast(Any, codegen_node))
    graph.add_node("execute", cast(Any, execute_node))
    graph.add_node("reflect", cast(Any, reflect_node))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "plan")
    graph.add_edge("plan", "codegen")
    graph.add_conditional_edges("codegen", route_after_codegen)
    graph.add_conditional_edges("execute", route_after_execute)
    graph.add_conditional_edges("reflect", route_after_reflect)

    return graph


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
                prompt = (
                    task_data.get("query")
                    or task_data.get("user_prompt")
                    or task_data.get("task")
                    or ""
                )
                initial_state = generate_default_state(task_id=task_id, user_prompt=prompt)

                async with semaphore:
                    try:
                        configurable = {
                            "thread_id": generate_thread_id(prefix=task_id),
                            "pipeline_params": pipeline_params.model_dump(),
                            "static_params": static_params.model_dump(),
                            "agent_params": agent_params.model_dump(),
                            "build_rag": asdict(rag_build_config),
                        }
                        final_state = await pipeline.ainvoke(
                            initial_state,
                            config=RunnableConfig(
                                configurable=configurable,
                                metadata=configurable
                            )
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
