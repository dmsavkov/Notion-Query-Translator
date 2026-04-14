from unittest.mock import patch

import pytest

from evaluation.test_title_search import make_title_search_target


@pytest.mark.unit
@pytest.mark.asyncio
async def test_title_search_target_uses_precheck_required_resources_for_resolution():
    target = make_title_search_target()
    inputs = {
        "task_id": "ts_case",
        "input_state": {
            "user_prompt": "Summarize AI Research",
        },
    }

    with patch(
        "evaluation.test_title_search.precheck_general_node",
        return_value={
            "meta": {
                "required_resources": ["AI Research"],
            }
        },
    ), patch(
        "evaluation.test_title_search.notion_requesting.search_pages_by_title",
        return_value=[{"id": "id-ai", "title": "AI Research", "properties": {}}],
    ):
        output = await target(inputs)

    assert output["task_id"] == "ts_case"
    assert output["inferred_required_resources"] == ["AI Research"]
    assert output["resource_map"] == {"AI Research": "id-ai"}
    assert output["mentioned_pages_count"] == 1
    assert output["resolved_pages_count"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_title_search_target_allows_empty_required_resources_from_precheck():
    target = make_title_search_target()
    inputs = {
        "task_id": "ts_missing",
        "input_state": {
            "user_prompt": "Summarize AI Research",
        },
    }

    with patch(
        "evaluation.test_title_search.precheck_general_node",
        return_value={
            "meta": {
                "required_resources": [],
            }
        },
    ) as mock_precheck:
        output = await target(inputs)

    assert output["task_id"] == "ts_missing"
    assert output["terminal_status"] == ""
    assert output["inferred_required_resources"] == []
    assert output["resource_map"] == {}
    mock_precheck.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_title_search_target_passes_precheck_resources_into_resolution_state():
    target = make_title_search_target()
    inputs = {
        "task_id": "ts_forwarding",
        "input_state": {
            "user_prompt": "Summarize AI Research",
        },
    }

    captured_state = {}

    async def fake_resolve_resources_node(state, config):
        captured_state.update(state)
        return {"resource_map": {"AI Research": "id-ai"}}

    with patch(
        "evaluation.test_title_search.precheck_general_node",
        return_value={
            "meta": {
                "required_resources": ["AI Research"],
                "reasoning": "precheck found one page",
            }
        },
    ), patch(
        "evaluation.test_title_search.resolve_resources_node",
        side_effect=fake_resolve_resources_node,
    ), patch(
        "evaluation.test_title_search.notion_requesting.search_pages_by_title",
        return_value=[{"id": "id-ai", "title": "AI Research", "properties": {}}],
    ):
        output = await target(inputs)

    assert output["task_id"] == "ts_forwarding"
    assert captured_state["meta"]["required_resources"] == ["AI Research"]
    assert captured_state["meta"]["reasoning"] == "precheck found one page"
    assert output["inferred_required_resources"] == ["AI Research"]
    assert output["resource_map"] == {"AI Research": "id-ai"}
