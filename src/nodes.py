"""
LangGraph Node implementations for the pipeline.

These nodes handle retrieval, planning, code generation, execution, and reflection steps.
"""

from functools import partial
from typing import Any, Dict, List, Literal, cast

from langchain_core.runnables import RunnableConfig

from .all_functionality import (
    async_chat_wrapper,
    build_general_info,
    generate_code,
    generate_request_plan,
    reflect_code,
)
from .guards import run_general_check, run_llama_guard_check
from .models.hardcoded_contexts import get_hardcoded_context
from .utils.execution_utils import run_isolated_code


# ── Query Helper Functions ────────────────────────────────────────────────────────────

async def _create_queries(
    query_method: Literal["multi_query", "cot_decompose", "domain_decompose"],
    engineer: Any,
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

async def precheck_general_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    user_prompt = state["user_prompt"]
    configurable = config.get("configurable", {})
    agent_params = configurable["agent_params"]
    precheck_params = agent_params.precheck
    if not bool(precheck_params.enabled):
        return {
            "meta": {
                "reasoning": "precheck disabled",
                "relevant_to_notion_scope": True,
                "complexity_label": "simple",
                "request_type": "UNKNOWN",
            }
        }

    meta = await run_general_check(
        query=user_prompt,
        model_name=str(precheck_params.general.model_name),
        model_temperature=float(precheck_params.general.model_temperature),
        max_tokens=precheck_params.general.max_tokens,
    )
    return {"meta": meta}


async def precheck_security_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    user_prompt = state["user_prompt"]
    configurable = config.get("configurable", {})
    agent_params = configurable["agent_params"]
    precheck_params = agent_params.precheck
    if not bool(precheck_params.enabled):
        return {
            "security": {
                "is_safe": True,
                "verdict": "safe",
                "violations": [],
                "raw": "precheck disabled",
                "error": "",
            }
        }

    security = await run_llama_guard_check(
        query=user_prompt,
        model_name=str(precheck_params.security.model_name),
        base_url=str(precheck_params.security.base_url),
        api_key_env=str(precheck_params.security.api_key_env),
        max_tokens=precheck_params.security.max_tokens,
    )
    return {"security": security}


async def precheck_join_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    return {}


async def malovolent_request_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    meta = state["meta"]
    security = state["security"]
    feedback = (
        "Blocked malovolent request. "
        f"reasoning={meta['reasoning']}; "
        f"verdict={security['verdict']}; "
        f"violations={security['violations']}"
    )
    print(feedback)
    return {"execution_output": feedback, "feedback": feedback}

async def retrieve_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    configurable = config.get("configurable", {})
    static_params = configurable["static_params"]
    qdrant_client = configurable.get("qdrant_client")
    context_used = str(static_params.context_used)

    if context_used != "dynamic":
        retrieval_context = get_hardcoded_context(context_used)
        return {"retrieval_context": retrieval_context, "queries": []}

    user_prompt = state.get("user_prompt", "")
    assert user_prompt, "Missing or empty 'user_prompt' in state before retrieval."
    
    agent_params = configurable["agent_params"]
    qt_params = agent_params.query_translator
    top_k = int(qt_params.top_k)
    top_k_total = int(qt_params.top_k_total)
    use_summarization = bool(qt_params.use_summarization)
    query_method_raw = str(qt_params.query_method)
    model_name = str(qt_params.model_name)
    model_temperature = float(qt_params.model_temperature)
    max_tokens = int(qt_params.max_tokens or 700)
    n_queries = int(qt_params.n_queries)

    if query_method_raw not in {"multi_query", "cot_decompose", "domain_decompose"}:
        query_method_raw = "cot_decompose"
    query_method = cast(Literal["multi_query", "cot_decompose", "domain_decompose"], query_method_raw)

    # Lazy import keeps static-context runs and unit tests independent from qdrant import-time issues.
    from .utils.rag_utils import (
        QueryEngineer,
        query_qdrant,
        search_multiple_queries,
        summarize_retrieval_results,
    )

    query_chat_fn = partial(
        async_chat_wrapper,
        model_size=model_name,
        temperature=model_temperature,
        max_tokens=max_tokens,
    )
    engineer = QueryEngineer(chat_fn=query_chat_fn, n_queries=n_queries)
    queries = await _create_queries(query_method, engineer, user_prompt)

    async def _search(query: str) -> List[Any]:
        return await query_qdrant(
            qdrant_client=qdrant_client,
            query=query,
            collection_name="notion_docs_leaf",
            top_k=top_k,
            threshold=0.2,
        )

    results = await search_multiple_queries(queries=queries, search_fn=_search)
    results = results[:top_k_total]
    
    if use_summarization:
        summarization_model_name = str(qt_params.summarization_model_name)
        summarization_temperature = float(qt_params.summarization_temperature)
        summarization_max_tokens = int(qt_params.summarization_max_tokens)
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
    assert state.get("user_prompt"), "Missing 'user_prompt' in state before planning."
    # Retrieval context may legitimately be empty, but the key must exist.
    assert "retrieval_context" in state, "Missing 'retrieval_context' in state before planning."
    
    configurable = config.get("configurable", {})
    static_params = configurable["static_params"]
    enable_planning = bool(static_params.enable_planning)

    if not enable_planning:
        # Planning disabled: build general_info without LLM planning
        request_plan = ""
        general_info = build_general_info(state["user_prompt"], state["retrieval_context"], request_plan)
        return {"request_plan": request_plan, "general_info": general_info}

    # Planning enabled: run full planning
    agent_params = configurable["agent_params"]
    rp_params = agent_params.request_planner
    model_name = str(rp_params.model_name)
    model_temperature = float(rp_params.model_temperature)
    max_tokens = int(rp_params.max_tokens or 1000)

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
    assert state.get("user_prompt"), "Missing 'user_prompt' in state before code generation."
    assert state.get("general_info"), "Missing 'general_info' in state before code generation."
    
    configurable = config.get("configurable", {})
    agent_params = configurable["agent_params"]
    cg_params = agent_params.code_generator
    model_size = str(cg_params.model_name)
    temperature = float(cg_params.model_temperature)
    max_tokens = int(cg_params.max_tokens or 2500)

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
    return {
        "generated_code": generated_code,
        "function_name": function_name,
        "trial_num": state.get("trial_num", 0) + 1,
    }


async def execute_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    assert state.get("task_id"), "Missing 'task_id' in state before execution."
    # generated_code may be empty from upstream failure; run_isolated_code handles it as a failed execution.
    assert "generated_code" in state, "Missing 'generated_code' in state before execution."

    # Execute the generated code using the centralized isolation utility
    result = run_isolated_code(
        code=str(state.get("generated_code") or ""),
        task_id=state["task_id"]
    )
    
    # Store both the raw result and the flattened output for easy access
    return {
        "solution_run": result.model_dump(),
        "execution_output": result.stdout if result.passed else f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    }


async def reflect_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    assert state.get("general_info"), "Missing 'general_info' in state before reflection."
    # generated_code can be empty on failed generation; reflector should still produce actionable feedback.
    assert "generated_code" in state, "Missing 'generated_code' key in state before reflection."

    configurable = config.get("configurable", {})
    agent_params = configurable["agent_params"]
    ref_params = agent_params.reflector
    model_size = str(ref_params.model_name)
    temperature = float(ref_params.model_temperature)
    max_tokens = int(ref_params.max_tokens or 1200)

    solution_run = state.get("solution_run") or {"exit_code": None, "stdout": "", "stderr": "", "passed": False}
    test_results = {
        "exit_code": solution_run["exit_code"],
        "stdout": solution_run["stdout"],
        "stderr": solution_run["stderr"],
        "passed": solution_run["passed"],
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
        "trials": [*state.get("trials", []), trial],
        "reflection_context": reflection_context,
        "final_code": state.get("generated_code", ""),
    }
