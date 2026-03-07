import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, cast
import datetime

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

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

@dataclass
class RunConfig:
    evals_dir: str = "evals"
    max_trials: int = 3
    minimal: bool = True
    output_dir: str = "evaluation_results"
    sqlite_saver_path = "data/checkpoints.sqlite"


class PipelineState(TypedDict):
    user_prompt: str
    max_trials: int
    minimal: bool
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


def route_after_codegen(state: PipelineState) -> str:
    if state.get("minimal", False):
        return END
    return "execute"


def route_after_reflect(state: PipelineState) -> str:
    if state.get("passed", False):
        return END
    if state.get("trial_num", 0) >= state.get("max_trials", 1):
        return END
    return "codegen"


def build_pipeline():
    graph = StateGraph(PipelineState)
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


async def main(cfg: RunConfig) -> Dict[str, Dict[str, Any]]:
    eval_tasks = load_eval_tasks(cfg.evals_dir)
    
    async with AsyncSqliteSaver.from_conn_string(cfg.sqlite_saver_path) as checkpointer:
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
                "user_prompt": prompt,
                "max_trials": cfg.max_trials,
                "minimal": cfg.minimal,
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

            final_state = await pipeline.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": generate_thread_id(prefix=task_id)}},
            )
            passed = bool(final_state.get("passed", False))
            results[task_id] = {
                "passed": passed,
                "final_code": final_state.get("final_code", ""),
                "trials": final_state.get("trials", []),
            }
            print(f"{task_id}: {'PASS' if passed else 'FAIL'}")
            break

        return results


if __name__ == "__main__":
    asyncio.run(main(RunConfig()))
