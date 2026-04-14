import asyncio
import json
import warnings
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from langchain_core.runnables import RunnableConfig

from src.evaluation.shared import evaluation_orchestration
from src.evaluation.utils import StandardEvaluationSettings, build_node_eval_state
from src.models.schema import AgentParams
from src.nodes import precheck_general_node, resolve_resources_node
from src.presentation import notion_requesting
from src.presentation import ui_bridge


SETTINGS = StandardEvaluationSettings(
	experiment_prefix="Title Search Node Evaluation",
	dataset_name="title_search_golden_dataset_v1",
	eval_max_concurrency=5,
	run_error_analysis_after_eval=False,
	evals_dir="evals",
	evals_case_type="title_search_v1.yaml",
	provision_infrastructure=True,
)


def _build_config(agent_params: Optional[AgentParams] = None) -> RunnableConfig:
	return RunnableConfig(configurable={"agent_params": agent_params or AgentParams()})


def _normalize_required_resources(value: Any) -> List[str]:
	if isinstance(value, str):
		cleaned = value.strip()
		return [cleaned] if cleaned else []

	if not isinstance(value, (list, tuple, set)):
		return []

	normalized: List[str] = []
	seen: set[str] = set()
	for item in value:
		cleaned = str(item).strip()
		if not cleaned or cleaned in seen:
			continue
		seen.add(cleaned)
		normalized.append(cleaned)
	return normalized


def _normalize_resource_map(value: Any) -> Dict[str, str]:
	if not isinstance(value, dict):
		return {}

	normalized: Dict[str, str] = {}
	for key, raw_id in value.items():
		title = str(key).strip()
		page_id = str(raw_id).strip()
		if title and page_id:
			normalized[title] = page_id
	return normalized


def _normalize_expected_titles(value: Any) -> List[str]:
	return _normalize_required_resources(value)


def _normalize_title(value: Any) -> str:
	return notion_requesting.normalize_title_for_match(value)


def _normalize_title_list(value: Any) -> List[str]:
	items = _normalize_required_resources(value)
	normalized: List[str] = []
	seen: set[str] = set()
	for item in items:
		title = _normalize_title(item)
		if not title or title in seen:
			continue
		seen.add(title)
		normalized.append(title)
	return normalized


def _normalize_search_observations(value: Any) -> Dict[str, List[Dict[str, str]]]:
	if not isinstance(value, dict):
		return {}

	normalized: Dict[str, List[Dict[str, str]]] = {}
	for key, row in value.items():
		title = str(key).strip()
		if not title:
			continue
		if not isinstance(row, list):
			normalized[title] = []
			continue

		items: List[Dict[str, str]] = []
		for item in row:
			if not isinstance(item, dict):
				continue
			item_id = str(item.get("id") or "").strip()
			item_title = str(item.get("title") or "").strip()
			if not item_id and not item_title:
				continue
			items.append({"id": item_id, "title": item_title})

		normalized[title] = items

	return normalized


def _candidate_titles_by_query(value: Any) -> Dict[str, List[str]]:
	observations = _normalize_search_observations(value)
	return {
		query_title: [str(item.get("title") or "").strip() for item in rows if str(item.get("title") or "").strip()]
		for query_title, rows in observations.items()
	}


def _resolve_title_by_id(page_id: str, observations: Dict[str, List[Dict[str, str]]], query_title: str) -> str:
	for item in observations.get(query_title, []):
		if str(item.get("id") or "").strip() == page_id:
			resolved = str(item.get("title") or "").strip()
			if resolved:
				return resolved

	for rows in observations.values():
		for item in rows:
			if str(item.get("id") or "").strip() == page_id:
				resolved = str(item.get("title") or "").strip()
				if resolved:
					return resolved

	return query_title


def make_title_search_target(agent_params: Optional[AgentParams] = None):
	node_config = _build_config(agent_params)

	async def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
		state = build_node_eval_state(inputs)
		task_id = str(state.get("task_id") or inputs.get("task_id") or "").strip()

		try:
			precheck_result = await precheck_general_node(state, node_config)
			meta = precheck_result.get("meta") if isinstance(precheck_result, dict) else {}
			meta = meta if isinstance(meta, dict) else {}
			required_titles = _normalize_required_resources(meta.get("required_resources"))

			state["meta"] = dict(meta)
			state["meta"]["required_resources"] = required_titles

			search_observations: Dict[str, List[Dict[str, str]]] = {}

			def _search_with_capture(title: str, limit: int = 10) -> List[Dict[str, Any]]:
				results = notion_requesting.search_pages_by_title(title=title, limit=int(limit))
				captured: List[Dict[str, str]] = []
				for item in results:
					if not isinstance(item, dict):
						continue
					item_id = str(item.get("id") or "").strip()
					captured.append({
						"id": item_id,
						"title": notion_requesting.extract_result_title(item),
					})
				search_observations[str(title).strip()] = captured
				return results

			async def _pick_first_disambiguation(title: str, options: List[Dict[str, Any]]) -> str:
				if not options:
					return ui_bridge.DISAMBIGUATION_CANCELLED

				normalized_title = _normalize_title(title)
				for option in options:
					option_id = str(option.get("id") or "").strip()
					if not option_id:
						continue
					if _normalize_title(option.get("title")) == normalized_title:
						return option_id

				fallback = str(options[0].get("id") or "").strip()
				return fallback or ui_bridge.DISAMBIGUATION_CANCELLED

			previous_disambiguator = ui_bridge.disambiguator
			with patch("src.nodes.search_pages_by_title", side_effect=_search_with_capture):
				ui_bridge.disambiguator = _pick_first_disambiguation
				try:
					result = await resolve_resources_node(state, node_config)
				finally:
					ui_bridge.disambiguator = previous_disambiguator

			resource_map = _normalize_resource_map(result.get("resource_map") if isinstance(result, dict) else {})
			resolved_titles = {
				query_title: _resolve_title_by_id(page_id, search_observations, query_title)
				for query_title, page_id in resource_map.items()
			}
			terminal_status = str(result.get("terminal_status") or "") if isinstance(result, dict) else ""
			execution_output = str(result.get("execution_output") or "") if isinstance(result, dict) else ""

			return {
				"task_id": task_id,
				"inferred_required_resources": required_titles,
				"search_observations": search_observations,
				"resource_map": resource_map,
				"resolved_titles": resolved_titles,
				"mentioned_pages_count": len(required_titles),
				"resolved_pages_count": len(resource_map),
				"terminal_status": terminal_status,
				"execution_output": execution_output,
			}
		except Exception as exc:
			warnings.warn(
				f"Title search node failed for task_id='{task_id}': {exc}",
				stacklevel=2,
			)
			return {
				"task_id": task_id,
				"inferred_required_resources": [],
				"search_observations": {},
				"resource_map": {},
				"resolved_titles": {},
				"mentioned_pages_count": 0,
				"resolved_pages_count": 0,
				"terminal_status": "execution_failed",
				"execution_output": str(exc),
				"error": str(exc),
			}

	return target


async def top1_precision_evaluator(
	*,
	inputs: Dict[str, Any],
	outputs: Dict[str, Any],
	reference_outputs: Dict[str, Any],
) -> Dict[str, Any]:
	expected_titles = _normalize_expected_titles(reference_outputs.get("required_resources"))
	candidate_titles = _candidate_titles_by_query(outputs.get("search_observations"))

	checked = 0
	matched = 0
	details: List[Dict[str, Any]] = []

	for expected_title in expected_titles:
		checked += 1
		normalized_expected = _normalize_title(expected_title)
		matched_queries: List[str] = []
		observed_top1: Dict[str, str] = {}
		for query_title, titles in candidate_titles.items():
			top1 = titles[0] if titles else ""
			if top1:
				observed_top1[query_title] = top1
			if top1 and _normalize_title(top1) == normalized_expected:
				matched_queries.append(query_title)

		is_match = bool(matched_queries)
		if is_match:
			matched += 1

		details.append(
			{
				"expected_title": expected_title,
				"matched_queries": matched_queries,
				"observed_top1": observed_top1,
				"match": is_match,
			}
		)

	score = 1.0 if checked == 0 else matched / float(checked)
	task_id = str(outputs.get("task_id") or reference_outputs.get("task_id") or inputs.get("task_id") or "")
	return {
		"key": "top1_precision",
		"score": score,
		"comment": json.dumps({"task_id": task_id, "checked": checked, "matched": matched, "details": details}),
	}


async def top3_recall_evaluator(
	*,
	inputs: Dict[str, Any],
	outputs: Dict[str, Any],
	reference_outputs: Dict[str, Any],
) -> Dict[str, Any]:
	expected_titles = _normalize_expected_titles(reference_outputs.get("required_resources"))
	candidate_titles = _candidate_titles_by_query(outputs.get("search_observations"))

	checked = 0
	matched = 0 
	details: List[Dict[str, Any]] = []

	for expected_title in expected_titles:
		checked += 1
		normalized_expected = _normalize_title(expected_title)
		matched_queries: List[str] = []
		observed_top3: Dict[str, List[str]] = {}
		for query_title, titles in candidate_titles.items():
			top3_titles = titles[:3]
			if top3_titles:
				observed_top3[query_title] = top3_titles
			if any(_normalize_title(top3_title) == normalized_expected for top3_title in top3_titles):
				matched_queries.append(query_title)

		is_match = bool(matched_queries)
		if is_match:
			matched += 1

		details.append(
			{
				"expected_title": expected_title,
				"matched_queries": matched_queries,
				"observed_top3": observed_top3,
				"match": is_match,
			}
		)

	score = 1.0 if checked == 0 else matched / float(checked)
	task_id = str(outputs.get("task_id") or reference_outputs.get("task_id") or inputs.get("task_id") or "")
	return {
		"key": "top3_recall",
		"score": score,
		"comment": json.dumps({"task_id": task_id, "checked": checked, "matched": matched, "details": details}),
	}


async def precheck_mention_count_evaluator(
	*,
	inputs: Dict[str, Any],
	outputs: Dict[str, Any],
	reference_outputs: Dict[str, Any],
) -> Dict[str, Any]:
	expected_titles = _normalize_title_list(reference_outputs.get("required_resources"))
	predicted_mentions = _normalize_title_list(outputs.get("inferred_required_resources"))
	predicted_count = len(predicted_mentions)
	predicted_resolved_count = int(outputs.get("resolved_pages_count") or 0)

	score = 1.0 if predicted_count == len(expected_titles) else 0.0
	task_id = str(outputs.get("task_id") or reference_outputs.get("task_id") or inputs.get("task_id") or "")
	return {
		"key": "precheck_mention_count_match",
		"score": score,
		"comment": json.dumps(
			{
				"task_id": task_id,
				"expected_referenced_pages_count": len(expected_titles),
				"predicted_mentioned_pages_count": predicted_count,
				"predicted_mentioned_pages": predicted_mentions,
				"predicted_resolved_pages_count": predicted_resolved_count,
			}
		),
	}


async def run_title_search_evaluation(settings: Optional[StandardEvaluationSettings] = None) -> Dict[str, Any]:
	final_settings = settings or SETTINGS

	return await evaluation_orchestration(
		settings=final_settings,
		target=make_title_search_target(),
		evaluators=[
			top1_precision_evaluator,
			top3_recall_evaluator,
			precheck_mention_count_evaluator,
		],
		human_config=None,
	)


@pytest.mark.asyncio
@pytest.mark.evaluation
async def test_title_search_evaluation() -> None:
	await run_title_search_evaluation()


async def main() -> None:
	await run_title_search_evaluation()


if __name__ == "__main__":
	asyncio.run(main())
