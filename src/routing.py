from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from .models.schema import PipelineState


def route_after_codegen(state: PipelineState, config: RunnableConfig) -> str:
    configurable = config.get("configurable", {})
    pipeline_params = configurable.get("pipeline_params")
    execution_method = str(getattr(pipeline_params, "execution_method", "local"))
    if execution_method == "sandbox":
        return "execute_sandbox"
    return "execute_local"


def route_after_precheck(state: PipelineState, config: RunnableConfig) -> str:
    meta = state["meta"]
    security = state["security"]
    if meta["relevant_to_notion_scope"] and security["is_safe"]:
        return "resolve_resources"
    return "malovolent_request"

def route_after_egress(state: PipelineState, config: RunnableConfig) -> str:
    pipeline_params = config.get("configurable", {}).get("pipeline_params")
    if str(state.get("terminal_status") or "") in {"security_blocked", "max_retries_exceeded"}:
        return END
    if bool(getattr(pipeline_params, "minimal", False)):
        return END
    return "reflect"


def route_after_reflect(state: PipelineState, config: RunnableConfig) -> str:
    pipeline_params = config.get("configurable", {}).get("pipeline_params")
    if str(state.get("terminal_status") or "") in {"success", "max_retries_exceeded"}:
        return END
    if state.get("trial_num", 0) >= int(getattr(pipeline_params, "max_trials", 0) or 0):
        return END
    return "codegen"
