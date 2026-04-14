from unittest.mock import AsyncMock, patch

import pytest

from src.guards import run_general_check


@pytest.mark.asyncio
async def test_run_general_check_normalizes_required_resources_to_lowercase_and_spaces() -> None:
    with patch("src.guards.async_chat_wrapper", new_callable=AsyncMock, return_value="raw-response") as mock_chat, patch(
        "src.guards.extract_json_from_response",
        return_value={
            "reasoning": "ok",
            "relevant_to_notion_scope": True,
            "required_resources": ["Critical_Overflow", "ID UPDATE PGE", "Blocked_by_New_Problem"],
        },
    ):
        result = await run_general_check("Find the pages")

    assert result["required_resources"] == ["critical overflow", "id update pge", "blocked by new problem"]
    mock_chat.assert_awaited_once()
