"""
LangGraph Node implementations for the pipeline.

These nodes handle retrieval, planning, code generation, execution, and reflection steps.
"""

import subprocess
import sys
from typing import Any, Dict, List

from .all_functionality import (
    async_chat_wrapper,
    build_general_info,
    generate_code,
    generate_request_plan,
    reflect_code,
    write_solution,
    qdrant_client,
)
from .config import SearchResult
from .rag_utils import (
    QueryEngineer,
    query_qdrant,
    search_multiple_queries,
    summarize_retrieval_results,
)


# ── LangGraph Node Wrappers ───────────────────────────────────────────────────────────

async def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = state["user_prompt"]
    engineer = QueryEngineer(chat_fn=async_chat_wrapper)
    decomposed = await engineer.domain_decompose(user_prompt)
    queries = [user_prompt] + decomposed

    async def _search(query: str) -> List[SearchResult]:
        return await query_qdrant(
            client=qdrant_client,
            query=query,
            collection_name="notion_docs_leaf",
            top_k=5,
            threshold=0.2,
        )

    results = await search_multiple_queries(queries=queries, search_fn=_search)
    retrieval_context = await summarize_retrieval_results(results, chat_fn=async_chat_wrapper)
    return {"retrieval_context": retrieval_context}


async def plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    request_plan = await generate_request_plan(state["user_prompt"], state["retrieval_context"])
    general_info = build_general_info(state["user_prompt"], state["retrieval_context"], request_plan)
    return {"request_plan": request_plan, "general_info": general_info}


async def codegen_node(state: Dict[str, Any]) -> Dict[str, Any]:
    feedback = state.get("feedback")
    code_result = await generate_code(
        general_info=state["general_info"],
        test_code="No tests are used.",
        feedback=feedback,
    )
    generated_code = code_result.get("code", "")
    function_name = code_result.get("function_name", "")
    write_solution(generated_code)
    return {
        "generated_code": generated_code,
        "function_name": function_name,
        "trial_num": state.get("trial_num", 0) + 1,
    }


async def execute_node(state: Dict[str, Any]) -> Dict[str, Any]:
    sol_run = subprocess.run(
        [sys.executable, "solution.py"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "solution_run": {
            "exit_code": sol_run.returncode,
            "stdout": sol_run.stdout,
            "stderr": sol_run.stderr,
        }
    }


async def reflect_node(state: Dict[str, Any]) -> Dict[str, Any]:
    solution_run = state.get("solution_run") or {"exit_code": None, "stdout": "", "stderr": ""}
    test_results = {
        "exit_code": solution_run["exit_code"],
        "stdout": solution_run["stdout"],
        "stderr": solution_run["stderr"],
        "passed": solution_run["exit_code"] == 0,
    }

    reflection_context = list(state.get("reflection_context", []))
    verdict = await reflect_code(
        general_info=state["general_info"],
        generated_code=state["generated_code"],
        test_results=test_results,
        solution_run=solution_run,
        reflection_context=reflection_context,
    )

    trial = {
        "trial_num": state.get("trial_num", 1),
        "code": state.get("generated_code", ""),
        "function_name": state.get("function_name", ""),
        "solution_run": solution_run,
        "verdict": verdict,
        "reflection_context": list(reflection_context),
    }
    return {
        "feedback": verdict.get("feedback", ""),
        "verdict": verdict,
        "passed": bool(verdict.get("pass", False)),
        "trials": [*state.get("trials", []), trial],
        "reflection_context": reflection_context,
        "final_code": state.get("generated_code", ""),
    }
