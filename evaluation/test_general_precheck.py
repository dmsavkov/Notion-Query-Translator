import asyncio
import json
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


def _normalize_required_resources(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []

    if not isinstance(value, (list, tuple, set)):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        cleaned = str(item).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return sorted(normalized)


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
                    "required_resources": [],
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
                "required_resources": _normalize_required_resources(meta.get("required_resources")),
            },
        }

    return target


async def required_resources_match_evaluator(
    *,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    expected = _normalize_required_resources(reference_outputs.get("required_resources"))

    predicted_meta = outputs.get("meta") if isinstance(outputs, dict) else {}
    if not isinstance(predicted_meta, dict):
        predicted_meta = {}
    predicted = _normalize_required_resources(predicted_meta.get("required_resources"))

    score = 1.0 if predicted == expected else 0.0
    task_id = str(outputs.get("task_id") or reference_outputs.get("task_id") or inputs.get("task_id") or "")
    return {
        "key": "required_resources_match",
        "score": score,
        "comment": json.dumps(
            {
                "task_id": task_id,
                "expected": expected,
                "predicted": predicted,
            }
        ),
    }


async def run_general_precheck_evaluation(settings: Optional[StandardEvaluationSettings] = None) -> Dict[str, Any]:
    final_settings = settings or SETTINGS
    evaluators = [
        ExactMatchEvaluator(
            keys_to_check=["relevant_to_notion_scope"],
            metric_key="relevant_to_notion_scope_match",
            output_container_key="meta",
        ),
    ]

    return await evaluation_orchestration(
        settings=final_settings,
        target=make_general_precheck_target(),
        evaluators=[evaluator.__call__ for evaluator in evaluators] + [required_resources_match_evaluator],
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