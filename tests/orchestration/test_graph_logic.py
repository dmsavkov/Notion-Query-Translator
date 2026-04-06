import pytest
from unittest.mock import AsyncMock, patch
from typing import Any, cast

from src.core.lifecycle import build_pipeline
from src.models.schema import (
    AgentParams,
    CodeGeneratorParams,
    PipelineParams,
    PrecheckParams,
    QueryTranslatorParams,
    ReflectorParams,
    RequestPlannerParams,
    StaticParams,
)
from src.nodes import (
    codegen_node,
    execute_node,
    plan_node,
    precheck_general_node,
    precheck_security_node,
    reflect_node,
    retrieve_node,
)
from src.routing import route_after_codegen, route_after_execute, route_after_reflect


def _make_agent_params(
    *,
    query_translator: QueryTranslatorParams | None = None,
    request_planner: RequestPlannerParams | None = None,
    code_generator: CodeGeneratorParams | None = None,
    reflector: ReflectorParams | None = None,
    precheck: PrecheckParams | None = None,
) -> AgentParams:
    return AgentParams(
        query_translator=query_translator or QueryTranslatorParams(),
        request_planner=request_planner or RequestPlannerParams(),
        code_generator=code_generator or CodeGeneratorParams(),
        reflector=reflector or ReflectorParams(),
        precheck=precheck or PrecheckParams(),
    )


def _config(
    *,
    pipeline: PipelineParams | None = None,
    static: StaticParams | None = None,
    agent: AgentParams | None = None,
    qdrant_client: object | None = None,
) -> Any:
    return {
        "configurable": {
            "thread_id": "test_thread",
            "pipeline_params": pipeline or PipelineParams(),
            "static_params": static or StaticParams(),
            "agent_params": agent or _make_agent_params(),
            "qdrant_client": qdrant_client,
        }
    }

@pytest.mark.orchestration
def test_route_after_codegen():
    # Codegen always routes to execute (no longer checks minimal mode)
    config_min = _config(pipeline=PipelineParams(minimal=True))
    assert route_after_codegen(cast(Any, {}), config_min) == "execute"
    
    config_full = _config(pipeline=PipelineParams(minimal=False))
    assert route_after_codegen(cast(Any, {}), config_full) == "execute"

@pytest.mark.orchestration
def test_route_after_execute():
    # Minimal mode: end after execute
    config_min = _config(pipeline=PipelineParams(minimal=True))
    assert route_after_execute(cast(Any, {}), config_min) == "__end__"
    
    # Full mode: continue to reflect
    config_full = _config(pipeline=PipelineParams(minimal=False))
    assert route_after_execute(cast(Any, {}), config_full) == "reflect"

@pytest.mark.orchestration
def test_route_after_reflect():
    # Success
    state_pass = cast(Any, {"verdict": {"pass": True}})
    config = _config(pipeline=PipelineParams(max_trials=3))
    assert route_after_reflect(state_pass, config) == "__end__"
    
    # Failure, trial 1
    state_fail_1 = cast(Any, {"verdict": {"pass": False}, "trial_num": 1})
    assert route_after_reflect(state_fail_1, config) == "codegen"
    
    # Failure, trial 3 (max reached)
    state_fail_3 = cast(Any, {"verdict": {"pass": False}, "trial_num": 3})
    assert route_after_reflect(state_fail_3, config) == "__end__"

@pytest.mark.orchestration
@pytest.mark.asyncio
async def test_full_graph_trajectory_mocked(mock_chat_wrapper):
    """
    Test a full graph trajectory where the first trial fails and the second succeeds.
    """
    # 1. Mock first codegen (failure)
    mock_chat_wrapper.side_effect = [
        # plan_node calls it (First call)
        "Step 1: Write code",
        # codegen_node trial 1 (Second call)
        {"code": "print(1/0)", "function_name": "main"},
        # reflect_node trial 1 (Third call)
        {"reasoning": "Division by zero", "pass": False, "feedback": "Fix division"},
        # codegen_node trial 2 (Fourth call)
        {"code": "print(42)", "function_name": "main"},
        # reflect_node trial 2 (Fifth call)
        {"reasoning": "Correct", "pass": True, "feedback": ""}
    ]
    
    pipeline = build_pipeline().compile()
    
    initial_state = {
        "task_id": "test_task",
        "user_prompt": "Do something",
        "retrieval_context": "",
        "request_plan": "",
        "general_info": "",
        "trial_num": 0,
        "generated_code": "",
        "function_name": "",
        "solution_run": {},
        "execution_output": "",
        "reflection_context": [],
        "feedback": "",
        "verdict": {},
        "trials": [],
        "final_code": "",
        "queries": [],
    }
    
    config = _config(
        pipeline=PipelineParams(minimal=False, max_trials=3),
        static=StaticParams(context_used="baseline", enable_planning=True),
        agent=_make_agent_params(
            query_translator=QueryTranslatorParams(
                top_k=3,
                top_k_total=5,
                use_summarization=False,
                query_method="cot_decompose",
                model_name="gemma4",
                model_temperature=0.3,
                max_tokens=500,
                n_queries=3,
            ),
            request_planner=RequestPlannerParams(model_name="gemma4", model_temperature=0.3, max_tokens=500),
            code_generator=CodeGeneratorParams(model_name="gemma4", model_temperature=0.3, max_tokens=500),
            reflector=ReflectorParams(model_name="gemma4", model_temperature=0.3, max_tokens=500),
        ),
    )
    
    final_state = await pipeline.ainvoke(initial_state, config=config)
    
    assert final_state["trial_num"] == 2
    assert final_state["verdict"]["pass"] is True
    assert len(final_state["trials"]) == 2


@pytest.mark.orchestration
def test_static_params_defaults_are_expected():
    params = StaticParams()
    assert params.case_type == "complex"
    assert params.enable_planning is False
    assert params.context_used != "dynamic"


@pytest.mark.orchestration
def test_agent_params_max_tokens_default_to_none():
    params = AgentParams()

    assert params.code_generator.max_tokens is None
    assert params.reflector.max_tokens is None
    assert params.query_translator.max_tokens is None
    assert params.request_planner.max_tokens is None
    assert params.precheck.general.max_tokens is None
    assert params.precheck.security.max_tokens is None


@pytest.mark.orchestration
@pytest.mark.asyncio
async def test_precheck_nodes_use_fallback_max_tokens_when_omitted():
    state = {"user_prompt": "Check this request"}
    config = _config(agent=AgentParams())

    with patch(
        "src.nodes.run_general_check",
        new_callable=AsyncMock,
        return_value={
            "reasoning": "ok",
            "relevant_to_notion_scope": True,
            "complexity_label": "simple",
            "request_type": "GET",
        },
    ) as mock_general, patch(
        "src.nodes.run_llama_guard_check",
        new_callable=AsyncMock,
        return_value={
            "is_safe": True,
            "verdict": "safe",
            "violations": [],
            "raw": "safe",
            "error": "",
        },
    ) as mock_security:
        general_result = await precheck_general_node(state, config)
        security_result = await precheck_security_node(state, config)

    assert general_result["meta"]["reasoning"] == "ok"
    assert security_result["security"]["verdict"] == "safe"
    assert mock_general.await_args is not None
    assert mock_general.await_args.kwargs["max_tokens"] == 700
    assert mock_security.await_args is not None
    assert mock_security.await_args.kwargs["max_tokens"] == 512


@pytest.mark.orchestration
@pytest.mark.asyncio
async def test_retrieve_node_uses_hardcoded_context_when_static():
    state = {"user_prompt": "ignored for static context"}
    config = _config(
        static=StaticParams(context_used="baseline", enable_planning=False),
        agent=_make_agent_params(),
    )

    with patch("src.nodes.get_hardcoded_context", return_value="STATIC_CTX") as mock_context:
        result = await retrieve_node(state, config)

    assert result["retrieval_context"] == "STATIC_CTX"
    assert result["queries"] == []
    mock_context.assert_called_once_with("baseline")


@pytest.mark.orchestration
@pytest.mark.asyncio
async def test_plan_node_skips_llm_when_planning_disabled():
    state = {
        "user_prompt": "Create a task",
        "retrieval_context": "Static Notion API context",
    }
    config = _config(
        static=StaticParams(enable_planning=False, context_used="baseline"),
        agent=_make_agent_params(
            request_planner=RequestPlannerParams(
                model_name="gemma4",
                model_temperature=0.91,
                max_tokens=777,
            )
        ),
    )

    with patch("src.nodes.generate_request_plan", new_callable=AsyncMock) as mock_planner:
        result = await plan_node(state, config)

    assert result["request_plan"] == ""
    assert "Create a task" in result["general_info"]
    assert "Static Notion API context" in result["general_info"]
    mock_planner.assert_not_awaited()


@pytest.mark.orchestration
@pytest.mark.asyncio
async def test_plan_node_passes_cfg_into_internal_chat_fn():
    state = {
        "user_prompt": "Create a task",
        "retrieval_context": "Static context",
    }
    config = _config(
        static=StaticParams(enable_planning=True, context_used="baseline"),
        agent=_make_agent_params(
            request_planner=RequestPlannerParams(
                model_name="gemma4",
                model_temperature=0.67,
                max_tokens=1333,
            )
        ),
    )

    with patch("src.nodes.generate_request_plan", new_callable=AsyncMock, return_value="Step 1") as mock_planner:
        result = await plan_node(state, config)

    assert result["request_plan"] == "Step 1"
    assert mock_planner.await_args is not None
    chat_fn = mock_planner.await_args.kwargs["chat_fn"]
    assert chat_fn.keywords["model_size"] == "gemma4"
    assert chat_fn.keywords["temperature"] == 0.67
    assert chat_fn.keywords["max_tokens"] == 1333


@pytest.mark.orchestration
@pytest.mark.asyncio
async def test_codegen_node_passes_cfg_to_generate_code():
    state = {
        "user_prompt": "Write code to solve the task",
        "general_info": "General info block",
        "feedback": "Fix header",
        "trial_num": 1,
    }
    config = _config(
        agent=_make_agent_params(
            code_generator=CodeGeneratorParams(
                model_name="gemma4",
                model_temperature=0.11,
                max_tokens=888,
            )
        ),
    )

    with patch(
        "src.nodes.generate_code",
        new_callable=AsyncMock,
        return_value={"code": "print('ok')", "function_name": "main"},
    ) as mock_codegen:
        result = await codegen_node(state, config)

    assert result["generated_code"] == "print('ok')"
    assert result["function_name"] == "main"
    assert result["trial_num"] == 2
    assert mock_codegen.await_args is not None
    assert mock_codegen.await_args.kwargs["model_size"] == "gemma4"
    assert mock_codegen.await_args.kwargs["temperature"] == 0.11
    assert mock_codegen.await_args.kwargs["max_tokens"] == 888


@pytest.mark.orchestration
@pytest.mark.asyncio
async def test_reflect_node_passes_cfg_and_records_trial():
    state = {
        "general_info": "General info block",
        "generated_code": "print('hello')",
        "function_name": "main",
        "solution_run": {"exit_code": 0, "stdout": "hello\n", "stderr": "", "passed": True},
        "reflection_context": ["prior-note"],
        "trials": [],
        "trial_num": 1,
    }
    config = _config(
        agent=_make_agent_params(
            reflector=ReflectorParams(
                model_name="gemma27",
                model_temperature=0.22,
                max_tokens=1444,
            )
        ),
    )

    verdict = {"reasoning": "Looks good", "pass": True, "feedback": ""}
    with patch("src.nodes.reflect_code", new_callable=AsyncMock, return_value=verdict) as mock_reflect:
        result = await reflect_node(state, config)

    assert result["verdict"] == verdict
    assert result["feedback"] == ""
    assert len(result["trials"]) == 1
    assert result["trials"][0]["verdict"]["pass"] is True
    assert result["final_code"] == "print('hello')"
    assert mock_reflect.await_args is not None
    assert mock_reflect.await_args.kwargs["model_size"] == "gemma27"
    assert mock_reflect.await_args.kwargs["temperature"] == 0.22
    assert mock_reflect.await_args.kwargs["max_tokens"] == 1444


@pytest.mark.orchestration
@pytest.mark.asyncio
async def test_execute_node_runs_code_and_captures_stdout():
    state = {
        "task_id": "stdout_case",
        "generated_code": "print('expected stdout')",
    }

    result = await execute_node(state, config={})

    assert result["solution_run"]["passed"] is True
    assert "expected stdout" in result["solution_run"]["stdout"]
    assert "expected stdout" in result["execution_output"]
