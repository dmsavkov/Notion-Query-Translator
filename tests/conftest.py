import pytest
import os
import sys
from unittest.mock import AsyncMock, patch

# Add project root to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def mock_chat_wrapper():
    """Fixture to mock async_chat_wrapper for orchestration tests."""
    with patch("src.nodes.async_chat_wrapper", new_callable=AsyncMock) as mock:
        yield mock

@pytest.fixture(scope="session")
def provisioned_env():
    """
    Fixture to provision the Notion test environment once per session.
    Only runs if NOTION_TOKEN is present in the environment.
    """
    if not os.getenv("NOTION_TOKEN"):
        pytest.skip("NOTION_TOKEN not found, skipping infrastructure provisioning")
    
    from evals.test_dbs_script import provision_infrastructure
    provision_infrastructure()
    return True

def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line("markers", "unit: fast, deterministic logic tests")
    config.addinivalue_line("markers", "orchestration: tests for graph trajectories and logic")
    config.addinivalue_line("markers", "smoke: basic connectivity and environment checks")
    config.addinivalue_line("markers", "llm: tests that perform real LLM calls (use with VCR)")
    config.addinivalue_line("markers", "asyncio: mark test as async (via pytest-asyncio)")

# pytest-asyncio default loop scope
@pytest.fixture(scope="session")
def event_loop_policy():
    import asyncio
    return asyncio.get_event_loop_policy()
