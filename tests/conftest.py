import pytest
import os
import sys
from unittest.mock import AsyncMock, patch

from notion_query.environment import load_runtime_environment


load_runtime_environment(include_sandbox=True)


@pytest.fixture
def mock_chat_wrapper():
    """Fixture to mock async_chat_wrapper for orchestration tests.
    
    Patches BOTH locations where async_chat_wrapper is referenced:
    1. src.all_functionality.async_chat_wrapper — where it's defined and directly called
    2. src.nodes.async_chat_wrapper — where it's imported and used with partial()
    
    Both patches must use the same mock object so side_effect synchronizes correctly.
    """
    # Create a single mock to be used in both locations
    shared_mock = AsyncMock()
    
    with patch("src.all_functionality.async_chat_wrapper", shared_mock):
        with patch("src.nodes.async_chat_wrapper", shared_mock):
            yield shared_mock

@pytest.fixture(scope="session")
def provisioned_env():
    """
    Fixture to provision the Notion test environment once per session.
    Only runs if NOTION_TOKEN is present in the environment.
    """
    if not os.getenv("NOTION_TOKEN"):
        pytest.skip("NOTION_TOKEN not found, skipping infrastructure provisioning")

    from src.evaluation.sandbox import provision_infrastructure

    provision_infrastructure()
    return True

def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line("markers", "unit: fast, deterministic logic tests")
    config.addinivalue_line("markers", "orchestration: tests for graph trajectories and logic")
    config.addinivalue_line("markers", "evaluation: tests that run LangSmith evaluation orchestration")
    config.addinivalue_line("markers", "smoke: basic connectivity and environment checks")
    config.addinivalue_line("markers", "integration: tests requiring real external services (E2B, Notion)")
    config.addinivalue_line("markers", "llm: tests that perform real LLM calls (use with VCR)")
    config.addinivalue_line("markers", "asyncio: mark test as async (via pytest-asyncio)")

# pytest-asyncio default loop scope
@pytest.fixture(scope="session")
def event_loop_policy():
    import asyncio
    return asyncio.get_event_loop_policy()
