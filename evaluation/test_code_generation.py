import asyncio
import json
import warnings
from typing import Any, Dict, List, Literal, Optional, cast

import pytest
from pydantic import ConfigDict

from src.all_functionality import build_general_info, load_eval_tasks
from src.evaluation.evaluator import Evaluator
from src.evaluation.shared import evaluation_orchestration, make_partial_live_eval_target
from src.evaluation.utils import StandardEvaluationSettings, _synthesize_eval_context, build_node_eval_state
from src.models.hardcoded_contexts import get_hardcoded_context
from src.models.schema import AgentParams, PipelineParams, RagBuildConfig, StaticParams
from src.presentation import notion_requesting


DEFAULT_CONTEXT_NAME = "database_schema_report_comprehensive__notion_api_top25_20220628"


class CodeGenerationEvaluationSettings(StandardEvaluationSettings):
	"""Settings for focused codegen->execute->reflect evaluation via partial graph execution."""

	judge_model_name: str = "gemini-3.1-flash-lite-preview"
	max_trials: int = 3
	execution_method: Literal["local", "sandbox"] = "sandbox"
	start_as_node: str = "plan"
	interrupt_before_nodes: tuple[str, ...] = ("cleanup_sandbox",)
	static_context_name: str = DEFAULT_CONTEXT_NAME

	model_config = ConfigDict(frozen=True)


SETTINGS = CodeGenerationEvaluationSettings(
	experiment_prefix="Codegen+Reflect Node Evaluation (Sandbox)",
	dataset_name="codegen_reflect_golden_dataset_v1",
	evals_case_type="codegen_reflect_v1.yaml",
	eval_max_concurrency=1,
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


def _error_target_output(task_id: str, error: str) -> Dict[str, Any]:
	return {
		"task_id": task_id,
		"thread_id": f"{task_id}_partial_live",
		"retrieval_context": "",
		"final_code": "",
		"execution": {},
		"solution_run": {},
		"execution_output": "",
		"function_name": "",
		"terminal_status": "execution_failed",
		"trial_num": 0,
		"error": error,
	}


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
	for raw_title, raw_id in value.items():
		title = str(raw_title).strip()
		page_id = str(raw_id).strip()
		if title and page_id:
			normalized[title] = page_id
	return normalized


def _required_resources_from_task_specs(task_id: str, task_specs: Dict[str, Dict[str, Any]]) -> List[str]:
	task_spec = task_specs.get(task_id)
	if not isinstance(task_spec, dict):
		return []

	input_state = task_spec.get("input_state")
	if isinstance(input_state, dict):
		input_titles = _normalize_required_resources(input_state.get("required_resources"))
		if input_titles:
			return input_titles

	reference_outputs = task_spec.get("reference_outputs")
	if isinstance(reference_outputs, dict):
		reference_titles = _normalize_required_resources(reference_outputs.get("required_resources"))
		if reference_titles:
			return reference_titles

	return []


async def _hydrate_resource_map_for_codegen(
	*,
	task_id: str,
	state: Dict[str, Any],
	task_specs: Dict[str, Dict[str, Any]],
) -> tuple[List[str], Dict[str, str]]:
	required_titles = _normalize_required_resources(state.get("required_resources"))
	if not required_titles:
		required_titles = _required_resources_from_task_specs(task_id, task_specs)

	resource_map = _normalize_resource_map(state.get("resource_map"))
	if not required_titles:
		return [], resource_map

	for title in required_titles:
		if title in resource_map:
			continue

		matches = await asyncio.to_thread(notion_requesting.search_pages_by_title, title=title, limit=10)
		if not matches:
			raise ValueError(f"Title-to-ID hydration failed: no Notion page matches '{title}'.")

		selected_id = notion_requesting.pick_exact_or_first_match_id(title, matches)
		if not selected_id:
			raise ValueError(f"Title-to-ID hydration failed: no valid page id found for '{title}'.")

		# Force a direct GET call to validate the resolved identifier.
		await asyncio.to_thread(notion_requesting.fetch_page_properties, selected_id)
		resource_map[title] = selected_id

	return required_titles, resource_map



def _load_static_context(context_name: str) -> str:
	try:
		return get_hardcoded_context(cast(Any, context_name))
	except Exception as exc:
		warnings.warn(
			f"Failed to load hardcoded context '{context_name}' for codegen eval: {exc}",
			stacklevel=2,
		)
		raise


async def _prepare_codegen_inputs(
	inputs: Dict[str, Any],
	settings: CodeGenerationEvaluationSettings,
	*,
	task_specs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
	task_specs = task_specs or {}
	state = build_node_eval_state(inputs)
	task_id = str(state.get("task_id") or inputs.get("task_id") or "").strip()
	user_prompt = str(state.get("user_prompt") or "").strip()

	if not user_prompt:
		raise ValueError("Missing user_prompt in eval sample input_state.")

	static_context_name = str(state.get("static_context_name") or settings.static_context_name or DEFAULT_CONTEXT_NAME).strip()
	if not static_context_name:
		raise ValueError("Missing static_context_name in eval sample input_state.")

	required_titles, hydrated_resource_map = await _hydrate_resource_map_for_codegen(
		task_id=task_id,
		state=state,
		task_specs=task_specs,
	)

	retrieval_context = str(state.get("retrieval_context") or "").strip() or _load_static_context(static_context_name)
	request_plan = ""

	general_info = str(state.get("general_info") or "").strip() or build_general_info(
		user_prompt=user_prompt,
		rag_context=retrieval_context,
		request_plan=request_plan,
	)

	original_input_state = inputs.get("input_state") if isinstance(inputs.get("input_state"), dict) else {}
	input_state = dict(cast(Dict[str, Any], original_input_state))
	input_state.update(
		{
			"task_id": task_id,
			"user_prompt": user_prompt,
			"static_context_name": static_context_name,
			"required_resources": required_titles,
			"resource_map": hydrated_resource_map,
			"retrieval_context": retrieval_context,
			"request_plan": request_plan,
			"general_info": general_info,
		}
	)
	if required_titles:
		meta = input_state.get("meta") if isinstance(input_state.get("meta"), dict) else {}
		meta["required_resources"] = required_titles
		input_state["meta"] = meta

	prepared = dict(inputs)
	prepared["input_state"] = input_state
	return prepared


def make_codegen_reflect_target(settings: CodeGenerationEvaluationSettings):
	task_specs = load_eval_tasks(settings.evals_dir, case_type=settings.evals_case_type)
	static_params = StaticParams(
		case_type="complex",
		context_used="dynamic",
		enable_planning=False,
		max_concurrency=1,
	)
	pipeline_params = PipelineParams(
		minimal=False,
		max_trials=int(settings.max_trials),
		execution_method=settings.execution_method,
		prompt_pass_sandbox_id_notion_pages=True,
		enable_page_caching=False,
	)
	base_target = make_partial_live_eval_target(
		static_params,
		pipeline_params,
		agent_params=AgentParams(),
		rag_build_config=RagBuildConfig(),
		start_as_node=settings.start_as_node,
		interrupt_before=list(settings.interrupt_before_nodes),
	)

	async def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
		task_id = str(inputs.get("task_id") or "").strip()
		try:
			prepared_inputs = await _prepare_codegen_inputs(
				inputs,
				settings,
				task_specs=task_specs,
			)
		except Exception as exc:
			warnings.warn(
				f"Codegen eval input preparation failed for task_id='{task_id}': {exc}",
				stacklevel=2,
			)
			return _error_target_output(task_id=task_id, error=str(exc))

		return await base_target(prepared_inputs)

	return target


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


def _build_evaluators(
	core_evaluator: Evaluator,
	settings: CodeGenerationEvaluationSettings,
) -> List[Any]:
	code_statements_eval = CodeStatementsEvaluator(core_evaluator, judge_model_name=settings.judge_model_name)
	code_execution_eval = CodeExecutionEvaluator(core_evaluator)
	return [
		code_statements_eval.__call__,
		code_execution_eval.__call__,
	]


async def run_code_generation_evaluation(
	settings: Optional[CodeGenerationEvaluationSettings] = None,
) -> Dict[str, Any]:
	final_settings = settings or SETTINGS
	core_evaluator = Evaluator(default_judge_model=final_settings.judge_model_name)

	target = make_codegen_reflect_target(final_settings)
	evaluators: List[Any] = _build_evaluators(core_evaluator, final_settings)

	return await evaluation_orchestration(
		settings=final_settings,
		target=target,
		evaluators=cast(Any, evaluators),
		human_config=None,
	)


@pytest.mark.asyncio
@pytest.mark.evaluation
async def test_code_generation_evaluation() -> None:
	await run_code_generation_evaluation()


async def main() -> None:
	await run_code_generation_evaluation()


if __name__ == "__main__":
	asyncio.run(main())
