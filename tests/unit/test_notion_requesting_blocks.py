from unittest.mock import patch

import pytest

from src.presentation.notion_requesting import resolve_block_to_page_id


@pytest.mark.unit
def test_resolve_block_to_page_id_direct_page_parent():
    with patch("src.presentation.notion_requesting.fetch_block", return_value={"parent": {"type": "page_id", "page_id": "page_123"}}):
        assert resolve_block_to_page_id("block_1") == "page_123"


@pytest.mark.unit
def test_resolve_block_to_page_id_walks_block_chain():
    def _fake_fetch(block_id: str):
        if block_id == "b1":
            return {"parent": {"type": "block_id", "block_id": "b2"}}
        if block_id == "b2":
            return {"parent": {"type": "page_id", "page_id": "page_999"}}
        raise AssertionError("Unexpected id")

    with patch("src.presentation.notion_requesting.fetch_block", side_effect=_fake_fetch):
        assert resolve_block_to_page_id("b1") == "page_999"


@pytest.mark.unit
def test_resolve_block_to_page_id_returns_none_when_unknown_parent():
    with patch("src.presentation.notion_requesting.fetch_block", return_value={"parent": {"type": "workspace"}}):
        assert resolve_block_to_page_id("block_x") is None

