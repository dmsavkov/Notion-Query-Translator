import json

import pytest

from src.adapters.cli_factory import build_app_config_from_cli
from src.models.schema import CliParams
from src.presentation.viewer import _prepare_completed_state


@pytest.mark.unit
def test_prepare_completed_state_promotes_single_page_stdout():
    raw_output = (
        "Imminent page found: 33ecb17d-cc44-816b-abac-dd71cadebdde\n"
        "{'object': 'page', 'id': '33ecb17d-cc44-816b-abac-dd71cadebdde', 'properties': {}}"
    )

    state = {
        "terminal_status": "success",
        "execution_output": raw_output,
        "affected_notion_ids": [],
    }

    prepared = _prepare_completed_state(state)

    assert prepared["affected_notion_ids"] == ["33ecb17d-cc44-816b-abac-dd71cadebdde"]
    assert prepared["execution_output"] == ""


@pytest.mark.unit
def test_prepare_completed_state_promotes_page_list_stdout():
    raw_output = (
        "[{'object': 'page', 'id': 'page-1', 'properties': {}}, "
        "{'object': 'page', 'id': 'page-2', 'properties': {}}]"
    )

    state = {
        "terminal_status": "success",
        "execution_output": raw_output,
        "affected_notion_ids": [],
    }

    prepared = _prepare_completed_state(state)

    assert prepared["affected_notion_ids"] == ["page-1", "page-2"]
    assert prepared["execution_output"] == ""


@pytest.mark.unit
def test_prepare_completed_state_leaves_non_page_stdout_untouched():
    state = {
        "terminal_status": "success",
        "execution_output": "hello world",
        "affected_notion_ids": [],
    }

    prepared = _prepare_completed_state(state)

    assert prepared == state


@pytest.mark.unit
def test_prepare_completed_state_parses_execution_envelope_and_caps_render_ids():
    payload = {
        "execution_status": "success",
        "message_to_user": "I found 7 tasks, showing the first 5.",
        "relevant_page_ids": ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
    }
    state = {
        "terminal_status": "success",
        "execution_output": f"debug line\n{json.dumps(payload)}",
        "affected_notion_ids": [],
    }

    prepared = _prepare_completed_state(state)

    assert prepared["execution_status"] == "success"
    assert prepared["message_to_user"] == payload["message_to_user"]
    assert prepared["execution_output"] == payload["message_to_user"]
    assert prepared["relevant_page_ids"] == payload["relevant_page_ids"]
    assert prepared["affected_notion_ids"] == ["p1", "p2", "p3", "p4", "p5"]


@pytest.mark.unit
def test_prepare_completed_state_caps_raw_page_list_stdout():
    raw_output = (
        "[{'object': 'page', 'id': 'page-1', 'properties': {}}, "
        "{'object': 'page', 'id': 'page-2', 'properties': {}}, "
        "{'object': 'page', 'id': 'page-3', 'properties': {}}]"
    )

    state = {
        "terminal_status": "success",
        "execution_output": raw_output,
        "affected_notion_ids": [],
    }

    prepared = _prepare_completed_state(state, render_cap=2)

    assert prepared["relevant_page_ids"] == ["page-1", "page-2", "page-3"]
    assert prepared["affected_notion_ids"] == ["page-1", "page-2"]
    assert prepared["execution_output"] == ""


@pytest.mark.unit
def test_build_app_config_from_cli_honors_render_cap():
    app_config = build_app_config_from_cli(
        cli_params=CliParams(
            user_prompt="Show me tasks",
            think=False,
            max_rendered_relevant_page_ids=12,
        )
    )

    assert app_config.pipeline.max_rendered_relevant_page_ids == 12