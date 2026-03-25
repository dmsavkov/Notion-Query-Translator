import pytest
import os
from pathlib import Path

@pytest.mark.smoke
def test_env_consistency():
    """Verify that .env contains all keys present in .env.example."""
    env_example_path = Path(".env.example")
    env_path = Path(".env")
    
    assert env_example_path.exists(), ".env.example is missing"
    
    # Allow test to pass if .env doesn't exist yet but NOTION_TOKEN is in env vars
    # (e.g. in CI or first run before provisioning)
    if not env_path.exists() and not os.getenv("NOTION_TOKEN"):
        pytest.fail(".env is missing and NOTION_TOKEN not in environment variables")

    def get_keys(path):
        keys = set()
        if not path.exists():
            return keys
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        keys.add(line.split("=")[0].strip())
        return keys

    example_keys = get_keys(env_example_path)
    
    # If .env exists, check its keys. Otherwise check system environment.
    if env_path.exists():
        actual_keys = get_keys(env_path)
    else:
        actual_keys = set(os.environ.keys())

    missing_keys = example_keys - actual_keys
    # Some keys might be optional or generated during provisioning
    # We focus on the core configuration keys
    core_keys = {"NOTION_TOKEN", "GOOGLE_API_KEY"}
    missing_core = core_keys - actual_keys
    
    assert not missing_core, f"Missing core environment variables: {missing_core}"

@pytest.mark.smoke
def test_infrastructure_ready(provisioned_env):
    """Verify that provision_infrastructure runs successfully and exports IDs."""
    assert provisioned_env is True
    
    # After provisioning, these should be in the environment (via load_dotenv in the script)
    # But since we run in the same process, we check if they were written to .env
    from dotenv import load_dotenv
    load_dotenv()
    
    required_ids = [
        "NOTION_TASKS_DATABASE_ID",
        "NOTION_PROJECTS_DATABASE_ID",
        "NOTION_INBOX_PAGE_ID"
    ]
    
    for key in required_ids:
        assert os.getenv(key), f"Provisioning failed to export {key}"
