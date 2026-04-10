import pytest
from unittest.mock import AsyncMock, patch
from src.nodes import resolve_resources_node

@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_resources_node_no_required_returns_empty_map():
    state = {"meta": {"required_resources": []}}
    config = {}
    result = await resolve_resources_node(state, config)
    assert result == {"resource_map": {}}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_resources_node_no_required_preserves_existing_map():
    state = {"meta": {"required_resources": []}, "resource_map": {"Known": "id-1"}}
    config = {}
    result = await resolve_resources_node(state, config)
    assert result == {"resource_map": {"Known": "id-1"}}

@pytest.mark.unit
@pytest.mark.asyncio
@patch("src.nodes.search_pages_by_title")
async def test_resolve_resources_node_single_match(mock_search):
    mock_search.return_value = [{"id": "notion-id-123"}]
    state = {"meta": {"required_resources": ["My Page"]}}
    config = {}
    
    result = await resolve_resources_node(state, config)
    assert result["resource_map"]["My Page"] == "notion-id-123"


@pytest.mark.unit
@pytest.mark.asyncio
@patch("src.nodes.search_pages_by_title")
async def test_resolve_resources_node_string_required_resource(mock_search):
    mock_search.return_value = [{"id": "notion-id-123"}]
    state = {"meta": {"required_resources": "My Page"}}
    config = {}

    result = await resolve_resources_node(state, config)
    assert result["resource_map"]["My Page"] == "notion-id-123"

@pytest.mark.unit
@pytest.mark.asyncio
@patch("src.nodes.search_pages_by_title")
async def test_resolve_resources_node_multiple_matches_no_disambiguator(mock_search):
    mock_search.return_value = [{"id": "id-1", "properties": {}}, {"id": "id-2", "properties": {}}]
    state = {"meta": {"required_resources": ["Ambig Page"]}}
    config = {}
    
    with patch("src.nodes.ui_bridge") as mock_bridge:
        mock_bridge.disambiguator = None
        result = await resolve_resources_node(state, config)
        
    assert result["terminal_status"] == "resource_not_found"
    assert "resource_map" in result
    assert "Ambiguity found" in result["execution_output"]

@pytest.mark.unit
@pytest.mark.asyncio
@patch("src.nodes.search_pages_by_title")
async def test_resolve_resources_node_multiple_matches_with_disambiguator(mock_search):
    mock_search.return_value = [{"id": "id-1", "properties": {}}, {"id": "id-2", "properties": {}}]
    state = {"meta": {"required_resources": ["Ambig Page"]}}
    config = {}
    
    mock_disambiguator = AsyncMock(return_value="id-2")
    
    with patch("src.nodes.ui_bridge") as mock_bridge:
        mock_bridge.disambiguator = mock_disambiguator
        result = await resolve_resources_node(state, config)
        
    assert result["resource_map"]["Ambig Page"] == "id-2"
    mock_disambiguator.assert_awaited_once()

@pytest.mark.unit
@pytest.mark.asyncio
@patch("src.nodes.search_pages_by_title")
async def test_resolve_resources_node_missing_page(mock_search):
    mock_search.return_value = []
    state = {"meta": {"required_resources": ["Non Existent"]}, "resource_map": {"Existing": "id-x"}}
    config = {}
    
    result = await resolve_resources_node(state, config)
    assert result["terminal_status"] == "resource_not_found"
    assert result["resource_map"] == {"Existing": "id-x"}
    assert "Could not find any Notion page matching the title" in result["execution_output"]
