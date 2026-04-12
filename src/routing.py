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
    meta = state.get("meta", {})
    security = state.get("security", {})
    if bool(meta.get("relevant_to_notion_scope")) and bool(security.get("is_safe")):
        return "resolve_resources"
    return "malovolent_request"


def route_after_plan(state: PipelineState, config: RunnableConfig) -> str | list[str]:
    pipeline_params = config.get("configurable", {}).get("pipeline_params")
    if str(getattr(pipeline_params, "execution_method", "local")) == "sandbox":
        return ["codegen", "prepare_sandbox"]
    return "codegen"


def route_after_resolve_resources(state: PipelineState, config: RunnableConfig) -> str:
    if str(state.get("terminal_status") or "") in {"resource_not_found", "ambiguity_unresolved", "execution_failed"}:
        return END
    return "retrieve"

def route_after_egress(state: PipelineState, config: RunnableConfig) -> str:
    pipeline_params = config.get("configurable", {}).get("pipeline_params")
    terminal_status = str(state.get("terminal_status") or "")
    if terminal_status in {"security_blocked", "max_retries_exceeded"}:
        return END
    if bool(getattr(pipeline_params, "minimal", False)):
        return END

    reflector_used = str(getattr(pipeline_params, "reflector_used", "self") or "self")
    if reflector_used == "none":
        return END
    if reflector_used == "external":
        return "reflect"

    # Self-correction mode: retry directly from execution failure without reflector.
    if terminal_status == "success":
        return END
    if terminal_status == "execution_failed":
        max_trials = int(getattr(pipeline_params, "max_trials", 0) or 0)
        if int(state.get("trial_num", 0) or 0) >= max_trials:
            return END
        return "codegen"

    return END


def route_after_reflect(state: PipelineState, config: RunnableConfig) -> str:
    pipeline_params = config.get("configurable", {}).get("pipeline_params")
    if str(getattr(pipeline_params, "reflector_used", "self") or "self") != "external":
        return END
    if str(state.get("terminal_status") or "") in {"success", "max_retries_exceeded"}:
        return END
    if state.get("trial_num", 0) >= int(getattr(pipeline_params, "max_trials", 0) or 0):
        return END
    return "codegen"
