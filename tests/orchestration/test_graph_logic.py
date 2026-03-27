import pytest
from unittest.mock import MagicMock
from langchain_core.runnables import RunnableConfig
from run_pipeline import build_pipeline, route_after_codegen, route_after_execute, route_after_reflect

@pytest.mark.orchestration
def test_route_after_codegen():
    # Codegen always routes to execute (no longer checks minimal mode)
    config_min = {"configurable": {"pipeline_params": {"minimal": True}}}
    assert route_after_codegen({}, config_min) == "execute"
    
    config_full = {"configurable": {"pipeline_params": {"minimal": False}}}
    assert route_after_codegen({}, config_full) == "execute"

@pytest.mark.orchestration
def test_route_after_execute():
    # Minimal mode: end after execute
    config_min = {"configurable": {"pipeline_params": {"minimal": True}}}
    assert route_after_execute({}, config_min) == "__end__"
    
    # Full mode: continue to reflect
    config_full = {"configurable": {"pipeline_params": {"minimal": False}}}
    assert route_after_execute({}, config_full) == "reflect"

@pytest.mark.orchestration
def test_route_after_reflect():
    # Success
    state_pass = {"verdict": {"pass": True}}
    config = {"configurable": {"pipeline_params": {"max_trials": 3}}}
    assert route_after_reflect(state_pass, config) == "__end__"
    
    # Failure, trial 1
    state_fail_1 = {"verdict": {"pass": False}, "trial_num": 1}
    assert route_after_reflect(state_fail_1, config) == "codegen"
    
    # Failure, trial 3 (max reached)
    state_fail_3 = {"verdict": {"pass": False}, "trial_num": 3}
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
    
    config = {
        "configurable": {
            "thread_id": "test_thread",
            "pipeline_params": {"minimal": False, "max_trials": 3},
            "static_params": {
                "context_used": "baseline", # Use hardcoded context
                "enable_planning": True
            },
            "agent_params": {
                "query_translator": {
                    "top_k": 3, "top_k_total": 5, "use_summarization": False, # Skip extra LLM call
                    "query_method": "cot_decompose", "model_name": "gemma4", 
                    "model_temperature": 0.3, "max_tokens": 500, "n_queries": 3
                },
                "request_planner": {"model_name": "gemma4", "model_temperature": 0.3, "max_tokens": 500},
                "code_generator": {"model_name": "gemma4", "model_temperature": 0.3, "max_tokens": 500},
                "reflector": {"model_name": "gemma4", "model_temperature": 0.3, "max_tokens": 500}
            }
        }
    }
    
    final_state = await pipeline.ainvoke(initial_state, config=config)
    
    assert final_state["trial_num"] == 2
    assert final_state["verdict"]["pass"] is True
    assert len(final_state["trials"]) == 2
