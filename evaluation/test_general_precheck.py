import asyncio
import warnings
from typing import Any, Dict, Optional

import pytest
from langchain_core.runnables import RunnableConfig

from src.evaluation.shared import ExactMatchEvaluator, evaluation_orchestration
from src.evaluation.utils import StandardEvaluationSettings, build_node_eval_state
from src.models.schema import AgentParams
from src.nodes import precheck_general_node


SETTINGS = StandardEvaluationSettings(
    experiment_prefix="General Precheck Node Evaluation",
    dataset_name="general_precheck_golden_dataset",
    eval_max_concurrency=5,
    run_error_analysis_after_eval=False,
    evals_dir="evals",
    evals_case_type="precheck/general_precheck_v1.yaml",
    provision_infrastructure=False,
)


def _build_config(agent_params: Optional[AgentParams] = None) -> RunnableConfig:
    return RunnableConfig(configurable={"agent_params": agent_params or AgentParams()})


def make_general_precheck_target(agent_params: Optional[AgentParams] = None):
    node_config = _build_config(agent_params)

    async def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        state = build_node_eval_state(inputs)
        try:
            result = await precheck_general_node(state, node_config)
        except Exception as exc:
            warnings.warn(
                f"General precheck node failed for task_id='{state.get('task_id') or ''}': {exc}",
                stacklevel=2,
            )
            return {
                "task_id": str(state.get("task_id") or inputs.get("task_id") or ""),
                "meta": {
                    "relevant_to_notion_scope": False,
                    "complexity_label": "UNKNOWN",
                    "request_type": "UNKNOWN",
                },
                "error": str(exc),
            }

        meta = result.get("meta") if isinstance(result, dict) else {}
        if not isinstance(meta, dict):
            meta = {}

        return {
            "task_id": str(state.get("task_id") or inputs.get("task_id") or ""),
            "meta": {
                "relevant_to_notion_scope": bool(meta.get("relevant_to_notion_scope", False)),
                "complexity_label": str(meta.get("complexity_label") or "UNKNOWN"),
                "request_type": str(meta.get("request_type") or "UNKNOWN").upper(),
            },
        }

    return target


async def run_general_precheck_evaluation(settings: Optional[StandardEvaluationSettings] = None) -> Dict[str, Any]:
    final_settings = settings or SETTINGS
    evaluators = [
        ExactMatchEvaluator(
            keys_to_check=["relevant_to_notion_scope"],
            metric_key="relevant_to_notion_scope_match",
            output_container_key="meta",
        ),
        ExactMatchEvaluator(
            keys_to_check=["complexity_label"],
            metric_key="complexity_label_match",
            output_container_key="meta",
        ),
        ExactMatchEvaluator(
            keys_to_check=["request_type"],
            metric_key="request_type_match",
            output_container_key="meta",
        ),
    ]

    return await evaluation_orchestration(
        settings=final_settings,
        target=make_general_precheck_target(),
        evaluators=[evaluator.__call__ for evaluator in evaluators],
        human_config=None,
    )


@pytest.mark.asyncio
@pytest.mark.evaluation
async def test_general_precheck_evaluation() -> None:
    await run_general_precheck_evaluation()


async def main() -> None:
    await run_general_precheck_evaluation()


if __name__ == "__main__":
    asyncio.run(main())