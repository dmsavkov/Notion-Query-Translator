"""LangGraph node implementations for the pipeline."""

import asyncio
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
from .presentation import ui_bridge
from .presentation.notion_requesting import search_pages_by_title
from langsmith import traceable


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
    return {
        "meta": meta,
    }


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


def _normalize_required_resources(raw_required: Any) -> List[str]:
    if isinstance(raw_required, str):
        value = raw_required.strip()
        return [value] if value else []

    if isinstance(raw_required, (list, tuple, set)):
        candidates = raw_required
    else:
        return []

    normalized: List[str] = []
    seen: set[str] = set()
    for item in candidates:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _extract_result_title(result: Dict[str, Any]) -> str:
    props = result.get("properties")
    if isinstance(props, dict):
        for prop_value in props.values():
            if not isinstance(prop_value, dict) or prop_value.get("type") != "title":
                continue
            title_items = prop_value.get("title")
            if not isinstance(title_items, list):
                continue
            text = "".join(str(item.get("plain_text", "")) for item in title_items if isinstance(item, dict)).strip()
            if text:
                return text
    return "Untitled"


@traceable(name="resolve_resources_node")
async def resolve_resources_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    meta = state.get("meta", {})
    required = _normalize_required_resources(meta.get("required_resources"))
    resource_map = dict(state.get("resource_map", {}))

    if not required:
        return {"resource_map": resource_map}
    
    for title in required:
        if title in resource_map:
            continue

        try:
            results = await asyncio.to_thread(search_pages_by_title, title=title, limit=10)
        except Exception as e:
            return {
                "terminal_status": "execution_failed",
                "execution_output": f"Failed to search Notion for '{title}': {e}",
                "resource_map": resource_map,
            }

        if not results:
            return {
                "terminal_status": "resource_not_found",
                "execution_output": f"Could not find any Notion page matching the title: '{title}'",
                "resource_map": resource_map,
            }

        if len(results) == 1:
            result_id = str(results[0].get("id", "")).strip()
            if not result_id:
                return {
                    "terminal_status": "execution_failed",
                    "execution_output": f"Notion search returned an item without an 'id' for '{title}'.",
                    "resource_map": resource_map,
                }
            resource_map[title] = result_id
            continue

        # Multiple results found - Disambiguate
        if ui_bridge.disambiguator is not None:
            # Interactive mode
            options = []
            for r in results:
                result_id = str(r.get("id", "")).strip()
                if not result_id:
                    continue
                options.append({"id": result_id, "title": _extract_result_title(r), "url": r.get("url")})

            if not options:
                return {
                    "terminal_status": "execution_failed",
                    "execution_output": f"Notion search returned ambiguous results without valid IDs for '{title}'.",
                    "resource_map": resource_map,
                }

            # Hand over to CLI
            try:
                selected_id = await ui_bridge.disambiguator(title, options)
            except Exception as e:
                return {
                    "terminal_status": "execution_failed",
                    "execution_output": f"Disambiguation failed for '{title}': {e}",
                    "resource_map": resource_map,
                }
            if not selected_id or selected_id == ui_bridge.DISAMBIGUATION_CANCELLED:
                return {
                    "terminal_status": "ambiguity_unresolved",
                    "execution_output": f"User cancelled disambiguation for '{title}'",
                    "resource_map": resource_map,
                }
            resource_map[title] = str(selected_id)
        else:
            # Non-interactive mode (tests/evals) - Fail with options
            option_titles = []
            for r in results:
                result_id = str(r.get("id", "")).strip()
                if not result_id:
                    continue
                option_titles.append(f"{_extract_result_title(r)} ({result_id})")
            
            return {
                "terminal_status": "resource_not_found",
                "execution_output": f"Ambiguity found for '{title}' (found {len(results)} matches) but no disambiguator available. Options: {', '.join(option_titles)}",
                "resource_map": resource_map,
            }

    return {"resource_map": resource_map}


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
    general_info_with_resources = state["general_info"]
    resource_map = state.get("resource_map", {})
    if resource_map:
        res_text = "\n".join([f"  - '{k}': {v}" for k, v in resource_map.items()])
        general_info_with_resources += f"\n\n<resource_map_context>\nThe following titles have already been resolved to Notion IDs. You MUST use these IDs when interacting with these specific resources. A variable named `RESOURCE_MAP` (dict) containing these mappings is injected into your global scope.\n{res_text}\n</resource_map_context>"

    code_result = await generate_code(
        general_info=general_info_with_resources,
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
    resource_map = state.get("resource_map", {})
    instrumented_code = wrap_code_with_telemetry(raw_code, local=True, resource_map=resource_map)
    result = await asyncio.to_thread(run_isolated_code, code=instrumented_code, task_id=state["task_id"])
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


@traceable(name="prepare_sandbox_node")
async def prepare_sandbox_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    configurable = config.get("configurable", {})
    pipeline_params = configurable.get("pipeline_params")
    if str(getattr(pipeline_params, "execution_method", "local")) != "sandbox":
        return {}

    terminal_status = str(state.get("terminal_status") or "")
    if terminal_status and terminal_status != "pending":
        return {}

    sandbox_id = state.get("sandbox_id")
    if sandbox_id:
        return {"sandbox_id": sandbox_id}

    thread_id = str(configurable.get("thread_id") or state["task_id"])
    template = str(getattr(pipeline_params, "sandbox_template", "notion-query-execution-sandbox"))
    client_timeout_seconds = int(getattr(pipeline_params, "sandbox_client_timeout_seconds", 300))

    try:
        sandbox = await asyncio.to_thread(
            get_or_create_sandbox,
            sandbox_id=None,
            thread_id=thread_id,
            task_id=state["task_id"],
            template=template,
            timeout_seconds=client_timeout_seconds,
        )
    except Exception:
        return {}

    return {"sandbox_id": getattr(sandbox, "sandbox_id", None)}


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
    resource_map = state.get("resource_map", {})
    instrumented_code = wrap_code_with_telemetry(raw_code, local=False, resource_map=resource_map)

    try:
        sandbox = await asyncio.to_thread(
            get_or_create_sandbox,
            sandbox_id=sandbox_id,
            thread_id=thread_id,
            task_id=state["task_id"],
            template=template,
            timeout_seconds=client_timeout_seconds,
        )
        sandbox_id = sandbox.sandbox_id

        result = await asyncio.to_thread(
            run_code_in_sandbox,
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
            file_bytes = await asyncio.to_thread(sandbox.files.read, AFFECTED_IDS_PATH)
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
        # Cleanup is awaited here so teardown stays deterministic; fire-and-forget would
        # return faster but could leave orphaned sandboxes if the event loop exits early.
        await asyncio.to_thread(kill_sandbox, sandbox_id)
    return {"sandbox_id": None}