from unittest.mock import patch

import pytest

from evaluation.test_code_generation import CodeGenerationEvaluationSettings, _prepare_codegen_inputs


@pytest.mark.unit
@pytest.mark.asyncio
async def test_prepare_codegen_inputs_hydrates_resource_map_from_input_titles():
    settings = CodeGenerationEvaluationSettings(
        evals_case_type="codegen_reflect_v1.yaml",
        provision_infrastructure=False,
    )
    inputs = {
        "task_id": "cg_task",
        "input_state": {
            "user_prompt": "Create a task related to AI Research",
            "required_resources": ["AI Research"],
        },
    }

    with patch("evaluation.test_code_generation._load_static_context", return_value="STATIC_CTX"), patch(
        "evaluation.test_code_generation.notion_requesting.search_pages_by_title",
        return_value=[{"id": "id-ai", "title": "AI Research"}],
    ) as mock_search, patch(
        "evaluation.test_code_generation.notion_requesting.fetch_page_properties",
        return_value={"id": "id-ai"},
    ) as mock_get:
        prepared = await _prepare_codegen_inputs(inputs, settings, task_specs={})

    prepared_state = prepared["input_state"]
    assert prepared_state["required_resources"] == ["AI Research"]
    assert prepared_state["resource_map"] == {"AI Research": "id-ai"}
    assert prepared_state["meta"]["required_resources"] == ["AI Research"]
    mock_search.assert_called_once_with(title="AI Research", limit=10)
    mock_get.assert_called_once_with("id-ai")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_prepare_codegen_inputs_uses_task_spec_titles_when_input_missing_titles():
    settings = CodeGenerationEvaluationSettings(
        evals_case_type="codegen_reflect_v1.yaml",
        provision_infrastructure=False,
    )
    inputs = {
        "task_id": "cg_task",
        "input_state": {
            "user_prompt": "Append checklist items to sandbox inbox",
        },
    }
    task_specs = {
        "cg_task": {
            "input_state": {
                "required_resources": ["Sandbox Inbox"],
            }
        }
    }

    with patch("evaluation.test_code_generation._load_static_context", return_value="STATIC_CTX"), patch(
        "evaluation.test_code_generation.notion_requesting.search_pages_by_title",
        return_value=[{"id": "id-inbox", "title": "Sandbox Inbox"}],
    ) as mock_search, patch(
        "evaluation.test_code_generation.notion_requesting.fetch_page_properties",
        return_value={"id": "id-inbox"},
    ) as mock_get:
        prepared = await _prepare_codegen_inputs(inputs, settings, task_specs=task_specs)

    prepared_state = prepared["input_state"]
    assert prepared_state["required_resources"] == ["Sandbox Inbox"]
    assert prepared_state["resource_map"] == {"Sandbox Inbox": "id-inbox"}
    mock_search.assert_called_once_with(title="Sandbox Inbox", limit=10)
    mock_get.assert_called_once_with("id-inbox")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_prepare_codegen_inputs_accepts_empty_required_resources_without_search_calls():
    settings = CodeGenerationEvaluationSettings(
        evals_case_type="codegen_reflect_v1.yaml",
        provision_infrastructure=False,
    )
    inputs = {
        "task_id": "cg_task",
        "input_state": {
            "user_prompt": "Find tasks due in exactly 3 days and set Intensity to 8.",
            "required_resources": [],
        },
    }

    with patch("evaluation.test_code_generation._load_static_context", return_value="STATIC_CTX"), patch(
        "evaluation.test_code_generation.notion_requesting.search_pages_by_title",
        return_value=[],
    ) as mock_search, patch(
        "evaluation.test_code_generation.notion_requesting.fetch_page_properties",
        return_value={},
    ) as mock_get:
        prepared = await _prepare_codegen_inputs(inputs, settings, task_specs={})

    prepared_state = prepared["input_state"]
    assert prepared_state["required_resources"] == []
    assert prepared_state["resource_map"] == {}
    mock_search.assert_not_called()
    mock_get.assert_not_called()
