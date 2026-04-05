import asyncio
import json
import warnings
from typing import Any, Dict, List, Optional, cast

import pytest

from pydantic import ConfigDict

from src.evaluation.shared import evaluation_orchestration, make_live_eval_target
from src.evaluation.utils import StandardEvaluationSettings, _synthesize_eval_context
from src.error_analysis import HumanConfig
from src.evaluation.evaluator import Evaluator
from src.models.schema import AgentParams, PipelineParams, RagBuildConfig, StaticParams


class E2EEvaluationSettings(StandardEvaluationSettings):
    judge_model_name: str = "gemini-3.1-flash-lite-preview"

    model_config = ConfigDict(frozen=True)


SETTINGS = E2EEvaluationSettings(
    experiment_prefix="COMPLEX CONTEXT UPDATED: personal comprehensive + top25_20220628 + scratch, refl3.",
    dataset_name="Dataset Complex v2.",
    evals_case_type="complex",
    eval_max_concurrency=5,
    run_error_analysis_after_eval=True,
    evals_dir="evals",
    provision_infrastructure=True,
)


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





