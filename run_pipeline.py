import asyncio
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, TypedDict, cast
import datetime

from pydantic import BaseModel, ConfigDict
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig

from build_rag import RagBuildConfig
from src.nodes import (
    codegen_node,
    execute_node,
    plan_node,
    reflect_node,
    retrieve_node,
)
from src.all_functionality import load_eval_tasks


def generate_thread_id(prefix: Optional[str] = None) -> str:
    right_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = f"{prefix}_{right_now}" if prefix else right_now
    return unique_id


class CodeGeneratorParams(BaseModel):
    model_name: str = "gemma27"
    model_temperature: float = 0.3
    max_tokens: int = 4000

    model_config = ConfigDict(frozen=True)


class ReflectorParams(BaseModel):
    model_name: str = "gemma27"
    model_temperature: float = 0.2
    max_tokens: int = 1200

    model_config = ConfigDict(frozen=True)


class QueryTranslatorParams(BaseModel):
    model_name: str = "gemma27"
    model_temperature: float = 0.3
    max_tokens: int = 1500
    top_k: int = 5
    top_k_total: int = 8
    query_method: Literal["multi_query", "cot_decompose", "domain_decompose"] = "cot_decompose"
    use_summarization: bool = False

    model_config = ConfigDict(frozen=True)


class AgentParams(BaseModel):
    """Per-node model and retrieval settings."""
    code_generator: CodeGeneratorParams = CodeGeneratorParams()
    reflector: ReflectorParams = ReflectorParams()
    query_translator: QueryTranslatorParams = QueryTranslatorParams()

    model_config = ConfigDict(frozen=True)


class PipelineParams(BaseModel):
    """Dynamic parameters used during pipeline execution."""
    minimal: bool = True
    max_trials: int = 3

    model_config = ConfigDict(frozen=True)


class StaticParams(BaseModel):
    """Static parameters for pipeline initialization."""
    evals_dir: str = "evals"
    output_dir: str = "evaluation_results"
    sqlite_saver_path: str = "data/checkpoints.sqlite"
    
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
    reflection_context: List[str]
    feedback: str
    verdict: Dict[str, Any]
    trials: List[Dict[str, Any]]
    final_code: str
    passed: bool


def route_after_codegen(state: PipelineState, config: Optional[RunnableConfig] = None) -> str:
    minimal = config.get("configurable", {}).get("pipeline_params", {}).get("minimal", False) if config else False
    if minimal:
        return END
    return "execute"


def route_after_reflect(state: PipelineState, config: Optional[RunnableConfig] = None) -> str:
    if state.get("passed", False):
        return END
    max_trials = config.get("configurable", {}).get("pipeline_params", {}).get("max_trials", 1) if config else 1
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
    graph.add_edge("execute", "reflect")
    graph.add_conditional_edges("reflect", route_after_reflect)

    return graph


async def main() -> Dict[str, Dict[str, Any]]:
    pipeline_params = PipelineParams()
    static_params = StaticParams()
    agent_params = AgentParams()
    rag_build_config = RagBuildConfig()
    
    eval_tasks = load_eval_tasks(static_params.evals_dir)
    
    async with AsyncSqliteSaver.from_conn_string(static_params.sqlite_saver_path) as checkpointer:
        pipeline = build_pipeline().compile(checkpointer=checkpointer)

        results: Dict[str, Dict[str, Any]] = {}
        for task_id, task_data in eval_tasks.items():
            prompt = (
                task_data.get("query")
                or task_data.get("user_prompt")
                or task_data.get("task")
                or ""
            )
            initial_state: PipelineState = {
                "task_id": task_id,
                "user_prompt": prompt,
                "retrieval_context": "",
                "request_plan": "",
                "general_info": "",
                "trial_num": 0,
                "generated_code": "",
                "function_name": "",
                "solution_run": {},
                "reflection_context": [],
                "feedback": "",
                "verdict": {},
                "trials": [],
                "final_code": "",
                "passed": False,
            }

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
                passed = bool(final_state.get("passed", False))
                results[task_id] = {
                    "passed": passed,
                    "final_code": final_state.get("final_code", ""),
                    "trials": final_state.get("trials", []),
                }
                print(f"{task_id}: {'PASS' if passed else 'FAIL'}")
            except Exception as e:
                print("Exception occured. ")
                print(e)

        return results


if __name__ == "__main__":
    asyncio.run(main())
