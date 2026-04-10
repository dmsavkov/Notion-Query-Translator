import pytest
from unittest.mock import AsyncMock, patch

from src.presentation import ui_bridge
from src.presentation.cli_helpers import cli_disambiguator


class _FakePrompt:
    def __init__(self, result):
        self.ask_async = AsyncMock(return_value=result)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cli_disambiguator_restores_bridge_state_on_selection():
    ui_bridge.current_node = "resolve_resources"

    with patch("src.presentation.cli_helpers.questionary.select", return_value=_FakePrompt("id-2")) as mock_select:
        result = await cli_disambiguator(
            "validation page",
            [{"id": "id-1", "title": "One", "url": "https://example.com/1"}],
        )

    assert result == "id-2"
    assert ui_bridge.current_node == "resolve_resources"
    mock_select.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cli_disambiguator_returns_cancel_sentinel_and_restores_state():
    ui_bridge.current_node = "resolve_resources"

    with patch("src.presentation.cli_helpers.questionary.select", return_value=_FakePrompt(ui_bridge.DISAMBIGUATION_CANCELLED)):
        result = await cli_disambiguator(
            "validation page",
            [{"id": "id-1", "title": "One", "url": "https://example.com/1"}],
        )

    assert result == ui_bridge.DISAMBIGUATION_CANCELLED
    assert ui_bridge.current_node == "resolve_resources"