from typing import Any, Dict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from .schema import PipelineState


def route_after_codegen(state: PipelineState, config: RunnableConfig) -> str:
    # Always execute at least once so pass/fail reflects real runtime behavior.
    return "execute"


def route_after_precheck(state: PipelineState, config: RunnableConfig) -> str:
    meta = state["meta"]
    security = state["security"]
    if meta["relevant_to_notion_scope"] and security["is_safe"]:
        return "retrieve"
    return "malovolent_request"


def route_after_execute(state: PipelineState, config: RunnableConfig) -> str:
    cfg = cast(Dict[str, Any], config.get("configurable", {}))
    pipeline_params = cast(Dict[str, Any], cfg.get("pipeline_params", {}))
    minimal = bool(pipeline_params.get("minimal", False))
    if minimal:
        return END
    return "reflect"


def route_after_reflect(state: PipelineState, config: RunnableConfig) -> str:
    # Use the LLM's pass/fail verdict for routing.
    if state.get("verdict", {}).get("pass", False):
        return END

    cfg = cast(Dict[str, Any], config.get("configurable", {}))
    pipeline_params = cast(Dict[str, Any], cfg.get("pipeline_params", {}))
    max_trials = int(pipeline_params.get("max_trials", 3))
    if state.get("trial_num", 0) >= max_trials:
        return END
    return "codegen"
