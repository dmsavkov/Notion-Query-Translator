import asyncio
import json
from pathlib import Path
import sys
import warnings
from typing import Any, Dict, List, Optional, cast
from unittest.mock import AsyncMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_utils import (
    evaluation_orchestration,
    StandardEvaluationSettings,
)
from src.core.lifecycle import run_with_lifecycle
from src.error_analysis import HumanConfig
from src.evaluator import Evaluator
from src.models.config import AppConfig
from src.models.schema import AgentParams, PipelineParams, RagBuildConfig, StaticParams

from pydantic import ConfigDict


class E2EEvaluationSettings(StandardEvaluationSettings):
    judge_model_name: str = "gemini-3.1-flash-lite-preview"

    model_config = ConfigDict(frozen=True)


SETTINGS = E2EEvaluationSettings(
    experiment_prefix="COMPLEX CONTEXT UPDATED: personal comprehensive + top25_20220628 + scratch, refl3.",
    evals_case_type="complex",
    eval_max_concurrency=5,
    run_error_analysis_after_eval=True,
    evals_dir="evals",
    provision_infrastructure=True,
)


def _synthesize_eval_context(
    *,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Synthesize all evaluation context from LangSmith parameters into a single dict.
    All values come from reference_outputs (ground truth) or outputs (predictions).

    Returns:
        {
            "task_id": str,
            "task": str,
            "solution": str,
            "retrieval_context": str,
            "final_code": str,
            "solution_run": Dict[str, Any],
            "execution_output": str,
            "correct_statements": List[str],
        }
    """
    task_id = (
        str(outputs.get("task_id") or "").strip()
        or str(reference_outputs.get("task_id") or "").strip()
        or str(inputs.get("task_id") or "").strip()
    )

    retrieval_context = str(outputs.get("retrieval_context") or "")
    final_code = str(outputs.get("final_code") or outputs.get("generated_code") or "")
    execution_output = str(outputs.get("execution_output") or "")

    solution_run = outputs.get("execution") or outputs.get("solution_run") or {}
    if not isinstance(solution_run, dict):
        solution_run = {}

    task = reference_outputs.get("task", "")
    solution = reference_outputs.get("solution", "")
    statements = reference_outputs.get("correct_statements") or []
    if not isinstance(statements, list):
        statements = []

    return {
        "task_id": task_id,
        "task": task,
        "solution": solution,
        "retrieval_context": retrieval_context,
        "final_code": final_code,
        "solution_run": solution_run,
        "execution_output": execution_output,
        "correct_statements": statements,
    }


def _error_target_output(task_id: str, thread_id: str, error: str) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "thread_id": thread_id,
        "retrieval_context": "",
        "final_code": "",
        "execution": {},
        "solution_run": {},
        "execution_output": "",
        "function_name": "",
        "error": error,
    }


def make_live_eval_target(
    static_params: StaticParams,
    pipeline_params: PipelineParams,
    *,
    agent_params: Optional[AgentParams] = None,
    rag_build_config: Optional[RagBuildConfig] = None,
):
    """
    Build a LangSmith target that executes one live pipeline run per dataset sample.

    The target delegates lifecycle ownership to run_with_lifecycle, which creates
    the SQL checkpointer and any conditional shared resources.
    """
    final_agent_params = agent_params or AgentParams()
    final_rag_build_config = rag_build_config or RagBuildConfig()
    app_config = AppConfig(
        pipeline=pipeline_params,
        static=static_params,
        agent=final_agent_params,
        rag=final_rag_build_config,
    )

    async def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        task_id = str(inputs.get("task_id") or "").strip()
        if not task_id:
            return _error_target_output(task_id="", thread_id="", error="Missing task_id in dataset inputs.")

        task_query = str(inputs.get("query") or inputs.get("user_prompt") or inputs.get("task") or "").strip()

        thread_id = f"{task_id}_live"
        if not task_query:
            return _error_target_output(
                task_id=task_id,
                thread_id=thread_id,
                error=f"No prompt found for task_id='{task_id}' in dataset inputs or task specs.",
            )

        try:
            result = await run_with_lifecycle(
                tasks={task_id: {"query": task_query}},
                app_config=app_config,
            )
            final_state = result[task_id]
        except Exception as exc:
            warnings.warn(
                f"Live target execution failed for task_id='{task_id}': {exc}",
                stacklevel=2,
            )
            return _error_target_output(task_id=task_id, thread_id=thread_id, error=str(exc))

        execution = final_state.get("solution_run") or {}
        if not isinstance(execution, dict):
            execution = {}

        return {
            "task_id": task_id,
            "thread_id": thread_id,
            "retrieval_context": str(final_state.get("retrieval_context") or ""),
            "final_code": str(final_state.get("final_code") or final_state.get("generated_code") or ""),
            "execution": execution,
            "solution_run": execution,
            "execution_output": str(final_state.get("execution_output") or ""),
            "function_name": str(final_state.get("function_name") or ""),
            "error": "",
        }

    return target


@pytest.mark.asyncio
async def test_live_eval_target_uses_lifecycle_and_maps_core_fields():
    target = make_live_eval_target(
        StaticParams(context_used="baseline"),
        PipelineParams(minimal=False),
    )

    final_state = {
        "task_id": "task-1",
        "retrieval_context": "ctx",
        "generated_code": "print(42)",
        "final_code": "",
        "solution_run": {"passed": True, "stdout": "42\n", "stderr": "", "exit_code": 0},
        "execution_output": "42\n",
        "function_name": "main",
    }

    with patch("evaluation.test_e2e.run_with_lifecycle", new_callable=AsyncMock, return_value={"task-1": final_state}) as mock_lifecycle:
        out = await target({"task_id": "task-1", "query": "Write code"})

    assert out["task_id"] == "task-1"
    assert out["retrieval_context"] == "ctx"
    assert out["final_code"] == "print(42)"
    assert out["execution"] == final_state["solution_run"]
    assert out["solution_run"] == final_state["solution_run"]
    assert out["execution_output"] == "42\n"
    assert out["error"] == ""
    mock_lifecycle.assert_awaited_once()
    assert mock_lifecycle.await_args is not None
    assert mock_lifecycle.await_args.kwargs["tasks"] == {"task-1": {"query": "Write code"}}


def _present_score(status_items: List[Dict[str, Any]]) -> tuple[float, int]:
    present_count = sum(
        1
        for item in status_items
        if str(item.get("status", "")).strip().lower() == "present"
    )
    score = (present_count / len(status_items)) if status_items else 0.0
    return score, present_count


class RagStatementsEvaluator:
    """Independent evaluator for statement presence in RAG retrieval context."""

    def __init__(self, evaluator: Evaluator, judge_model_name: str | None = None):
        self.evaluator = evaluator
        self.judge_model_name = judge_model_name

    async def _compute(
        self, *, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        ctx = _synthesize_eval_context(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
        )
        task_id = ctx["task_id"]
        retrieval_context = ctx["retrieval_context"]
        statements = ctx["correct_statements"]

        if not statements:
            warnings.warn(
                f"Task '{task_id}' has empty correct_statements for rag_statements_score.",
                stacklevel=2,
            )

        status_items = await self.evaluator.eval_context_statements(
            context=retrieval_context,
            statements=statements,
            judge_model_name=self.judge_model_name,
        )

        if not status_items:
            warnings.warn(
                f"rag_statements_score evaluator returned empty output for task '{task_id}'. Scoring as 0.0.",
                stacklevel=2,
            )
            return {
                "key": "rag_statements_score",
                "score": 0.0,
                "comment": json.dumps({"error": "Empty evaluator output", "task_id": task_id}),
            }

        score, _ = _present_score(status_items)

        return {
            "key": "rag_statements_score",
            "score": score,
            "comment": json.dumps(status_items),
        }

    async def __call__(
        self, *, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return await self._compute(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)


class CodeStatementsEvaluator:
    """Independent evaluator for statement presence in generated code."""

    def __init__(self, evaluator: Evaluator, judge_model_name: str | None = None):
        self.evaluator = evaluator
        self.judge_model_name = judge_model_name

    async def _compute(
        self, *, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        ctx = _synthesize_eval_context(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
        )
        task_id = ctx["task_id"]
        final_code = ctx["final_code"]
        statements = ctx["correct_statements"]

        if not statements:
            warnings.warn(
                f"Task '{task_id}' has empty correct_statements for code_statements_score.",
                stacklevel=2,
            )

        status_items = await self.evaluator.eval_context_statements(
            context=final_code,
            statements=statements,
            judge_model_name=self.judge_model_name,
        )

        if not status_items:
            warnings.warn(
                f"code_statements_score evaluator returned empty output for task '{task_id}'. Scoring as 0.0.",
                stacklevel=2,
            )
            return {
                "key": "code_statements_score",
                "score": 0.0,
                "comment": json.dumps({"error": "Empty evaluator output", "task_id": task_id}),
            }

        score, _ = _present_score(status_items)

        return {
            "key": "code_statements_score",
            "score": score,
            "comment": json.dumps(status_items),
        }

    async def __call__(
        self, *, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return await self._compute(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)


class CodeExecutionEvaluator:
    """Independent evaluator for generated code execution pass/fail."""

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

    async def _compute(
        self, *, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        ctx = _synthesize_eval_context(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
        )
        task_id = ctx["task_id"]

        result = await self.evaluator.eval_code_exec(
            execution=cast(Dict[str, Any], ctx.get("solution_run") or {}),
            execution_output=str(ctx.get("execution_output") or ""),
        )

        if not result:
            warnings.warn(
                f"code_execution_score evaluator returned empty output for task '{task_id}'. Scoring as 0.0.",
                stacklevel=2,
            )
            return {
                "key": "code_execution_score",
                "score": 0.0,
                "comment": json.dumps({"error": "Empty evaluator output", "task_id": task_id}),
            }

        execution_pass = bool(result.get("pass", False))

        return {
            "key": "code_execution_score",
            "score": 1.0 if execution_pass else 0.0,
            "comment": json.dumps(result),
        }

    async def __call__(
        self, *, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return await self._compute(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)


def _build_evaluators(core_evaluator: Evaluator, settings: E2EEvaluationSettings) -> List[Any]:
    rag_eval = RagStatementsEvaluator(core_evaluator, judge_model_name=settings.judge_model_name)
    code_statements_eval = CodeStatementsEvaluator(core_evaluator, judge_model_name=settings.judge_model_name)
    code_execution_eval = CodeExecutionEvaluator(core_evaluator)
    return [
        rag_eval.__call__,
        code_statements_eval.__call__,
        code_execution_eval.__call__,
    ]


async def run_e2e_evaluation(settings: Optional[E2EEvaluationSettings] = None) -> Dict[str, Any]:
    final_settings = settings or SETTINGS
    static_params = StaticParams(case_type=cast(Any, final_settings.evals_case_type))
    pipeline_params = PipelineParams(minimal=False)
    agent_params = AgentParams()
    rag_build_config = RagBuildConfig()
    core_evaluator = Evaluator(default_judge_model=final_settings.judge_model_name)

    target = make_live_eval_target(
        static_params,
        pipeline_params,
        agent_params=agent_params,
        rag_build_config=rag_build_config,
    )

    evaluators: List[Any] = _build_evaluators(core_evaluator, final_settings)

    return await evaluation_orchestration(
        settings=final_settings,
        target=target,
        evaluators=cast(Any, evaluators),
        human_config=HumanConfig(),
    )


@pytest.mark.asyncio
@pytest.mark.evaluation
async def test_e2e_live_evaluation() -> None:
    await run_e2e_evaluation()


async def main() -> None:
    await run_e2e_evaluation()


if __name__ == "__main__":
    asyncio.run(main())





