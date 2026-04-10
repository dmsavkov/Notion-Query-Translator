import pytest
import asyncio
from src.nodes import resolve_resources_node
from src.models.schema import generate_default_state
from langchain_core.runnables import RunnableConfig

@pytest.mark.asyncio
@pytest.mark.smoke
async def test_resolve_resources_real_notion():
    """Smoke test to verify real Notion API connection and resource resolution."""
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
    
    assert "resource_map" in result
    print(f"\nResolved resources: {result['resource_map']}")
    
    # If it found a match, resource_map should have an entry
    # If not found, it would have set terminal_status to resource_not_found
    if result.get("terminal_status") == "resource_not_found":
        print(f"Resource not found details: {result.get('execution_output')}")
    else:
        assert len(result["resource_map"]) > 0
