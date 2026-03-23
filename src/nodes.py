"""
LangGraph Node implementations for the pipeline.

These nodes handle retrieval, planning, code generation, execution, and reflection steps.
"""

import subprocess
import sys
from typing import Any, Dict, List, Literal, Optional
from functools import partial

from langchain_core.runnables import RunnableConfig

from .all_functionality import (
    async_chat_wrapper,
    build_general_info,
    generate_code,
    generate_request_plan,
    reflect_code,
    write_solution,
)
from .config import SearchResult
from .hardcoded_contexts import ContextUsed, get_hardcoded_context
from .rag_utils import (
    QueryEngineer,
    query_qdrant,
    search_multiple_queries,
    summarize_retrieval_results,
)


# ── Query Helper Functions ────────────────────────────────────────────────────────────

async def _create_queries(
    query_method: Literal["multi_query", "cot_decompose", "domain_decompose"],
    engineer: QueryEngineer,
    query: str,
) -> List[str]:
    """Build a list of queries based on the selected query method."""
    additional_queries: List[str] = []
    
    if query_method == "multi_query":
        additional_queries = await engineer.multi_query(query)
    elif query_method == "cot_decompose":
        additional_queries = await engineer.cot_decompose(query)
    elif query_method == "domain_decompose":
        additional_queries = await engineer.domain_decompose(query)
    
    queries = additional_queries + [query]
    return queries


# ── LangGraph Node Wrappers ───────────────────────────────────────────────────────────

async def retrieve_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    static_params = config["configurable"]["static_params"]
    context_used: ContextUsed = static_params["context_used"]

    if context_used != "dynamic":
        retrieval_context = get_hardcoded_context(context_used)
        return {"retrieval_context": retrieval_context, "queries": []}

    user_prompt = state["user_prompt"]
    qt_params = config["configurable"]["agent_params"]["query_translator"]
    top_k: int = qt_params["top_k"]
    top_k_total: int = qt_params["top_k_total"]
    use_summarization: bool = qt_params["use_summarization"]
    query_method: str = qt_params["query_method"]
    model_name: str = qt_params["model_name"]
    model_temperature: float = qt_params["model_temperature"]
    max_tokens: int = qt_params["max_tokens"]
    n_queries: int = qt_params["n_queries"]

    query_chat_fn = partial(
        async_chat_wrapper,
        model_size=model_name,
        temperature=model_temperature,
        max_tokens=max_tokens,
    )
    engineer = QueryEngineer(chat_fn=query_chat_fn, n_queries=n_queries)
    queries = await _create_queries(query_method, engineer, user_prompt)

    async def _search(query: str) -> List[SearchResult]:
        return await query_qdrant(
            query=query,
            collection_name="notion_docs_leaf",
            top_k=top_k,
            threshold=0.2,
        )

    results = await search_multiple_queries(queries=queries, search_fn=_search)
    results = results[:top_k_total]
    
    if use_summarization:
        summarization_model_name: str = qt_params["summarization_model_name"]
        summarization_temperature: float = qt_params["summarization_temperature"]
        summarization_max_tokens: int = qt_params["summarization_max_tokens"]
        summarize_chat_fn = partial(
            async_chat_wrapper,
            model_size=summarization_model_name,
            temperature=summarization_temperature,
            max_tokens=summarization_max_tokens,
        )
        retrieval_context = await summarize_retrieval_results(results, query=user_prompt, chat_fn=summarize_chat_fn)
    else:
        retrieval_context = "\n\n".join(r.text for r in results)
        
    return {"retrieval_context": retrieval_context, "queries": queries}


async def plan_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    static_params = config["configurable"]["static_params"]
    enable_planning: bool = static_params["enable_planning"]

    if not enable_planning:
        # Planning disabled: build general_info without LLM planning
        request_plan = ""
        general_info = build_general_info(state["user_prompt"], state["retrieval_context"], request_plan)
        return {"request_plan": request_plan, "general_info": general_info}

    # Planning enabled: run full planning
    rp_params = config["configurable"]["agent_params"]["request_planner"]
    model_name: str = rp_params["model_name"]
    model_temperature: float = rp_params["model_temperature"]
    max_tokens: int = rp_params["max_tokens"]

    plan_chat_fn = partial(
        async_chat_wrapper,
        model_size=model_name,
        temperature=model_temperature,
        max_tokens=max_tokens,
    )

    request_plan = await generate_request_plan(
        state["user_prompt"],
        state["retrieval_context"],
        chat_fn=plan_chat_fn,
    )
    general_info = build_general_info(state["user_prompt"], state["retrieval_context"], request_plan)
    return {"request_plan": request_plan, "general_info": general_info}


async def codegen_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    cg_params = config["configurable"]["agent_params"]["code_generator"]
    model_size: str = cg_params["model_name"]
    temperature: float = cg_params["model_temperature"]
    max_tokens: int = cg_params["max_tokens"]

    feedback = state.get("feedback")
    code_result = await generate_code(
        general_info=state["general_info"],
        test_code="No tests are used.",
        feedback=feedback,
        model_size=model_size,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    generated_code = code_result.get("code", "")
    function_name = code_result.get("function_name", "")
    write_solution(generated_code)
    return {
        "generated_code": generated_code,
        "function_name": function_name,
        "trial_num": state.get("trial_num", 0) + 1,
    }


async def execute_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
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


async def reflect_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    ref_params = config["configurable"]["agent_params"]["reflector"]
    model_size: str = ref_params["model_name"]
    temperature: float = ref_params["model_temperature"]
    max_tokens: int = ref_params["max_tokens"]

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
        model_size=model_size,
        temperature=temperature,
        max_tokens=max_tokens,
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
