import pytest
import asyncio
import os
from src.nodes import resolve_resources_node
from src.models.schema import generate_default_state
from langchain_core.runnables import RunnableConfig

@pytest.mark.asyncio
@pytest.mark.smoke
@pytest.mark.integration
async def test_resolve_resources_real_notion():
    """Smoke test to verify real Notion API connection and resource resolution."""
    if not os.getenv("NOTION_TOKEN"):
        pytest.skip("NOTION_TOKEN is required for real Notion smoke test")

    state = generate_default_state()
    state["user_prompt"] = "Get me the validation node page"
    # Note: we manually populate meta here to simulate precheck result
    state["meta"] = {"required_resources": ["Validation Node"]}
    
    config = RunnableConfig(configurable={})
    
    # This should now use asyncio.to_thread and not block the event loop
    try:
        result = await asyncio.wait_for(resolve_resources_node(state, config), timeout=60)
    except asyncio.TimeoutError:
        pytest.fail("resolve_resources_node timed out after 60 seconds")

    assert isinstance(result, dict)
    status = str(result.get("terminal_status") or "")

    if status in {"resource_not_found", "ambiguity_unresolved", "execution_failed"}:
        assert result.get("execution_output"), "Expected descriptive execution_output on resolver failure"
    else:
        assert "resource_map" in result
        assert isinstance(result["resource_map"], dict)
        assert len(result["resource_map"]) > 0
