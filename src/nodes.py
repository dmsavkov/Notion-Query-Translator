"""LangGraph node implementations for the pipeline."""

import json
import os
import sys
from contextlib import suppress
from functools import partial
from pathlib import Path
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
from .utils.execution_utils import (
    ExecutionResult,
    is_timeout_result,
    run_code_in_sandbox,
    run_isolated_code,
    get_or_create_sandbox,
    kill_sandbox,
)
from .utils.telemetry import (
    AFFECTED_IDS_PATH,
    LOCAL_AFFECTED_IDS_PATH,
    wrap_code_with_telemetry,
)


async def _create_queries(
    query_method: Literal["multi_query", "cot_decompose", "domain_decompose"],
    engineer: Any,
    query: str,
) -> List[str]:
    if query_method == "multi_query":
        additional_queries = await engineer.multi_query(query)
    elif query_method == "cot_decompose":
        additional_queries = await engineer.cot_decompose(query)
    else:
        additional_queries = await engineer.domain_decompose(query)
    return [*additional_queries, query]


async def precheck_general_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    configurable = config.get("configurable", {})
    precheck_params = configurable["agent_params"].precheck
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
        query=state["user_prompt"],
        model_name=str(precheck_params.general.model_name),
        model_temperature=float(precheck_params.general.model_temperature),
        max_tokens=precheck_params.general.max_tokens,
    )
    return {"meta": meta}


async def precheck_security_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    configurable = config.get("configurable", {})
    precheck_params = configurable["agent_params"].precheck
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
        query=state["user_prompt"],
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
    print(feedback, file=sys.stderr)
    return {
        "execution_output": feedback,
        "feedback": feedback,
        "terminal_status": "security_blocked",
    }

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
    query_method_raw = str(qt_params.query_method)
    if query_method_raw not in {"multi_query", "cot_decompose", "domain_decompose"}:
        query_method_raw = "cot_decompose"
    query_method = cast(Literal["multi_query", "cot_decompose", "domain_decompose"], query_method_raw)

    # Lazy import keeps static-context runs and unit tests independent from qdrant import-time issues.
    from .utils.rag_utils import QueryEngineer, query_qdrant, search_multiple_queries, summarize_retrieval_results

    query_chat_fn = partial(
        async_chat_wrapper,
        model_size=str(qt_params.model_name),
        temperature=float(qt_params.model_temperature),
        max_tokens=qt_params.max_tokens,
    )
    engineer = QueryEngineer(chat_fn=query_chat_fn, n_queries=int(qt_params.n_queries))
    queries = await _create_queries(query_method, engineer, user_prompt)

    async def _search(query: str) -> List[Any]:
        return await query_qdrant(
            qdrant_client=configurable.get("qdrant_client"),
            query=query,
            collection_name="notion_docs_leaf",
            top_k=int(qt_params.top_k),
            threshold=0.2,
        )

    results = (await search_multiple_queries(queries=queries, search_fn=_search))[: int(qt_params.top_k_total)]
    if bool(qt_params.use_summarization):
        summarize_chat_fn = partial(
            async_chat_wrapper,
            model_size=str(qt_params.summarization_model_name),
            temperature=float(qt_params.summarization_temperature),
            max_tokens=int(qt_params.summarization_max_tokens),
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
    if not bool(configurable["static_params"].enable_planning):
        request_plan = ""
        return {
            "request_plan": request_plan,
            "general_info": build_general_info(state["user_prompt"], state["retrieval_context"], request_plan),
        }

    request_plan = await generate_request_plan(
        state["user_prompt"],
        state["retrieval_context"],
        chat_fn=partial(
            async_chat_wrapper,
            model_size=str(configurable["agent_params"].request_planner.model_name),
            temperature=float(configurable["agent_params"].request_planner.model_temperature),
            max_tokens=configurable["agent_params"].request_planner.max_tokens,
        ),
    )
    return {
        "request_plan": request_plan,
        "general_info": build_general_info(state["user_prompt"], state["retrieval_context"], request_plan),
    }


async def codegen_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    assert state.get("user_prompt"), "Missing 'user_prompt' in state before code generation."
    assert state.get("general_info"), "Missing 'general_info' in state before code generation."

    configurable = config.get("configurable", {})
    code_result = await generate_code(
        general_info=state["general_info"],
        test_code="No tests are used.",
        feedback=state.get("feedback"),
        model_size=str(configurable["agent_params"].code_generator.model_name),
        temperature=float(configurable["agent_params"].code_generator.model_temperature),
        max_tokens=configurable["agent_params"].code_generator.max_tokens,
    )
    return {
        "generated_code": code_result.get("code", ""),
        "function_name": code_result.get("function_name", ""),
        "trial_num": state.get("trial_num", 0) + 1,
    }


def _execution_state_update(result: ExecutionResult) -> Dict[str, Any]:
    terminal_status = (
        "success" if result.passed else "max_retries_exceeded" if is_timeout_result(result) else "execution_failed"
    )
    return {
        "solution_run": result.model_dump(),
        "execution_output": result.stdout if result.passed else f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        "terminal_status": terminal_status,
    }


async def execute_local_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    assert state.get("task_id"), "Missing 'task_id' in state before execution."
    assert "generated_code" in state, "Missing 'generated_code' in state before execution."

    raw_code = str(state.get("generated_code") or "")
    instrumented_code = wrap_code_with_telemetry(raw_code, local=True)
    result = run_isolated_code(code=instrumented_code, task_id=state["task_id"])
    update_data = _execution_state_update(result)

    # Extract affected IDs from the local temp file
    affected_ids: List[str] = []
    with suppress(Exception):
        ids_path = Path(LOCAL_AFFECTED_IDS_PATH)
        if ids_path.exists():
            affected_ids = json.loads(ids_path.read_text())
            ids_path.unlink(missing_ok=True)
    update_data["affected_notion_ids"] = affected_ids
    return update_data


async def execute_sandbox_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    assert state.get("task_id"), "Missing 'task_id' in state before execution."
    assert "generated_code" in state, "Missing 'generated_code' in state before execution."

    configurable = config.get("configurable", {})
    pipeline_params = configurable.get("pipeline_params")
    thread_id = str(configurable.get("thread_id") or state["task_id"])
    template = str(getattr(pipeline_params, "sandbox_template", "notion-query-execution-sandbox"))
    client_timeout_seconds = int(getattr(pipeline_params, "sandbox_client_timeout_seconds", 300))
    execution_timeout_seconds = int(getattr(pipeline_params, "sandbox_execution_timeout_seconds", 15))
    sandbox = None
    sandbox_id = state.get("sandbox_id")

    raw_code = str(state.get("generated_code") or "")
    instrumented_code = wrap_code_with_telemetry(raw_code, local=False)

    try:
        sandbox = get_or_create_sandbox(
            sandbox_id=sandbox_id,
            thread_id=thread_id,
            task_id=state["task_id"],
            template=template,
            timeout_seconds=client_timeout_seconds,
        )
        sandbox_id = sandbox.sandbox_id

        result = run_code_in_sandbox(
            sandbox=sandbox,
            code=instrumented_code,
            execution_timeout_seconds=execution_timeout_seconds,
        )
    except Exception as exc:
        result = ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr=str(exc),
            passed=False,
            error=type(exc).__name__,
            metadata={"method": "sandbox", "thread_id": thread_id, "task_id": state["task_id"], "template": template},
        )

    # Extract affected IDs from the sandbox filesystem
    affected_ids: List[str] = []
    if sandbox is not None:
        with suppress(Exception):
            file_bytes = sandbox.files.read(AFFECTED_IDS_PATH)
            affected_ids = json.loads(file_bytes)

    update_data = _execution_state_update(result)
    update_data["sandbox_id"] = sandbox_id
    update_data["affected_notion_ids"] = affected_ids
    return update_data


async def execute_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    pipeline_params = config.get("configurable", {}).get("pipeline_params")
    if str(getattr(pipeline_params, "execution_method", "local")) == "sandbox":
        return await execute_sandbox_node(state, config)
    return await execute_local_node(state, config)


async def egress_security_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    output_text = str(state.get("execution_output") or "")
    pipeline_params = config.get("configurable", {}).get("pipeline_params")
    egress_checked_tokens = list(getattr(pipeline_params, "egress_checked_tokens", []) or [])

    for env_name in egress_checked_tokens:
        token_value = os.getenv(str(env_name), "")
        if token_value and token_value in output_text:
            return {
                "execution_output": "[SECURITY OVERRIDE - OUTPUT DELETED]",
                "terminal_status": "security_blocked",
            }

    return {}


async def reflect_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    assert state.get("general_info"), "Missing 'general_info' in state before reflection."
    assert "generated_code" in state, "Missing 'generated_code' key in state before reflection."

    configurable = config.get("configurable", {})
    ref_params = configurable["agent_params"].reflector
    solution_run = state.get("solution_run") or {"exit_code": None, "stdout": "", "stderr": "", "passed": False}
    verdict = await reflect_code(
        general_info=state["general_info"],
        generated_code=state["generated_code"],
        test_results={
            "exit_code": solution_run["exit_code"],
            "stdout": solution_run["stdout"],
            "stderr": solution_run["stderr"],
            "passed": solution_run["passed"],
        },
        solution_run=solution_run,
        reflection_context=list(state.get("reflection_context", [])),
        model_size=str(ref_params.model_name),
        temperature=float(ref_params.model_temperature),
        max_tokens=ref_params.max_tokens,
    )

    trial_num = int(state.get("trial_num", 0) or 0)
    max_trials = int(getattr(configurable.get("pipeline_params"), "max_trials", 0) or 0)
    return {
        "feedback": verdict.get("feedback", ""),
        "verdict": verdict,
        "trials": [
            *state.get("trials", []),
            {
                "trial_num": state.get("trial_num", 1),
                "code": state.get("generated_code", ""),
                "function_name": state.get("function_name", ""),
                "solution_run": solution_run,
                "verdict": verdict,
                "reflection_context": list(state.get("reflection_context", [])),
            },
        ],
        "reflection_context": list(state.get("reflection_context", [])),
        "final_code": state.get("generated_code", ""),
        "terminal_status": "success" if verdict.get("pass", False) else "max_retries_exceeded" if trial_num >= max_trials else "execution_failed",
    }


async def cleanup_sandbox_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    sandbox_id = state.get("sandbox_id")
    if sandbox_id:
        kill_sandbox(sandbox_id)
    return {"sandbox_id": None}