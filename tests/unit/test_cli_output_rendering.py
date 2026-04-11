import pytest

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