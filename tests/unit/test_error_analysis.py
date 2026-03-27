import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


_fake_all_functionality = types.ModuleType("src.all_functionality")


async def _fake_async_chat_wrapper(*args, **kwargs):
    return "stub"


_fake_all_functionality.async_chat_wrapper = _fake_async_chat_wrapper
sys.modules.setdefault("src.all_functionality", _fake_all_functionality)

from src.error_analysis import HumanConfig, _build_section_payloads, _statement_items, load_experiment_runs, run_error_analysis


def _sample_record() -> dict:
    return {
        "experiment": "exp_a",
        "task_id": "task_1",
        "thread_id": "thread_1",
        "run_id": "run_1",
        "scores": {
            "code_execution_score": 1.0,
            "code_statements_score": 0.5,
        },
        "final_code": "print('ok')",
        "retrieval_context": "static ctx",
        "comments": {
            "code_statements_score": [
                {"statement": "A", "status": "present", "reasoning": "ok", "evidence": "x"},
                {"statement": "B", "status": "wrong", "reasoning": "bad", "evidence": "y"},
            ]
        },
        "outputs": {
            "thread_id": "thread_1",
            "pre_computed_state": {
                "task_id": "task_1",
                "final_code": "print('ok')",
                "retrieval_context": "static ctx",
                "request_plan": "Step 1 -> Step 2",
            },
        },
    }


@pytest.mark.unit
def test_load_experiment_runs_extracts_complete_records():
    project = SimpleNamespace(name="exp_prefix_A")
    run = SimpleNamespace(
        id="run_123",
        outputs={
            "thread_id": "thread_123",
            "pre_computed_state": {
                "task_id": "task_123",
                "final_code": "print('hello')",
                "retrieval_context": "ctx",
            },
        },
    )
    feedback_rows = [
        SimpleNamespace(run_id="run_123", key="code_execution_score", score=1.0, comment=None),
        SimpleNamespace(
            run_id="run_123",
            key="code_statements_score",
            score=0.5,
            comment='[{"statement": "must have auth", "status": "wrong"}]',
        ),
    ]

    fake_client = MagicMock()
    fake_client.list_projects.return_value = [project, SimpleNamespace(name="other")]
    fake_client.list_runs.return_value = [run]
    fake_client.list_feedback.return_value = feedback_rows

    with patch("src.error_analysis.Client", return_value=fake_client):
        records = load_experiment_runs("exp_prefix", dataset_name="Dataset v4.")

    assert len(records) == 1
    record = records[0]
    assert record["experiment"] == "exp_prefix_A"
    assert record["task_id"] == "task_123"
    assert record["thread_id"] == "thread_123"
    assert record["final_code"] == "print('hello')"
    assert record["retrieval_context"] == "ctx"
    assert record["scores"]["code_execution_score"] == 1.0
    assert record["scores"]["code_statements_score"] == 0.5
    assert isinstance(record["comments"]["code_statements_score"], list)


@pytest.mark.unit
def test_statement_items_filters_status_and_limit():
    records = [_sample_record()]

    wrong_only = _statement_items(
        records,
        score_key="code_statements_score",
        status_filter="wrong",
        max_examples=1,
    )

    assert len(wrong_only) == 1
    assert wrong_only[0]["status"] == "wrong"
    assert wrong_only[0]["statement"] == "B"


@pytest.mark.unit
def test_build_section_payloads_produces_complete_outputs():
    records = [_sample_record()]
    cfg = HumanConfig(
        include_code=True,
        include_code_execution=True,
        include_code_statements=True,
        include_code_mismatches=True,
        include_rag=False,
        include_rag_statements=False,
        include_plans=False,
        include_all_in_one=False,
        judging_enabled=False,
        max_examples_per_field=2,
        max_code_chars=100,
    )

    with patch("src.error_analysis.build_group_report_prompt", side_effect=lambda artifacts: f"PROMPT::{artifacts[:20]}"):
        payloads = _build_section_payloads(records, cfg)

    enabled_payloads = [p for p in payloads if p.enabled]
    enabled_names = {p.section_name for p in enabled_payloads}

    assert enabled_names == {"code", "code_execution", "code_statements", "code_mismatches"}
    assert all(isinstance(p.output, dict) for p in enabled_payloads)
    assert all(p.prompt.startswith("PROMPT::") for p in enabled_payloads)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_error_analysis_end_to_end_minimal_flow():
    records = [_sample_record()]
    cfg = HumanConfig(
        include_code=True,
        include_code_statements=True,
        include_code_mismatches=True,
        judging_enabled=False,
        max_examples_per_field=1,
        max_code_chars=100,
        max_rag_chars=100,
        max_plan_chars=100,
    )

    append_mock = MagicMock()
    copy_mock = MagicMock()

    with patch("src.error_analysis.load_dotenv"), patch("src.error_analysis._ensure_model_aliases"), patch(
        "src.error_analysis.load_experiment_runs", return_value=records
    ), patch(
        "src.error_analysis.build_group_report_prompt", side_effect=lambda artifacts: f"PROMPT::{artifacts[:10]}"
    ), patch(
        "src.error_analysis.create_tracking_page", return_value="page_123"
    ), patch(
        "src.error_analysis._append_children", append_mock
    ), patch(
        "src.error_analysis.build_architecture_analysis_prompt", return_value="FINAL_PROMPT"
    ), patch(
        "src.error_analysis._copy_to_clipboard", copy_mock
    ):
        result = await run_error_analysis("exp_a", config=cfg, dataset_name="Dataset v4.")

    assert result["page_id"] == "page_123"
    assert result["record_count"] == 1
    assert result["clipboard_prompt"] == "FINAL_PROMPT"

    enabled_sections = [s for s in result["sections"] if s["enabled"]]
    assert len(enabled_sections) == 3
    assert all("output" in s and "judge_output" in s for s in enabled_sections)

    append_mock.assert_called_once()
    children = append_mock.call_args.args[1]
    raw_toggle_present = any(
        block.get("type") == "toggle"
        and block.get("toggle", {}).get("rich_text", [{}])[0].get("text", {}).get("content")
        == "Raw error analysis outputs"
        for block in children
    )
    assert raw_toggle_present

    copy_mock.assert_called_once_with("FINAL_PROMPT")
