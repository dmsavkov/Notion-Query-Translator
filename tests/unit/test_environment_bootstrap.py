import os
from pathlib import Path

import pytest

from notion_query.environment import initialize_runtime_environment, load_runtime_environment


@pytest.mark.unit
def test_load_runtime_environment_applies_sandbox_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base_env = tmp_path / ".env"
    sandbox_env = tmp_path / ".env.sandbox"

    base_env.write_text("NOTION_TOKEN=base-token\nNOTION_TASKS_DATABASE_ID=base-db\n", encoding="utf-8")
    sandbox_env.write_text("NOTION_TASKS_DATABASE_ID=sandbox-db\nNOTION_INBOX_PAGE_ID=sandbox-inbox\n", encoding="utf-8")

    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    monkeypatch.delenv("NOTION_TASKS_DATABASE_ID", raising=False)
    monkeypatch.delenv("NOTION_INBOX_PAGE_ID", raising=False)

    load_runtime_environment(
        base_env_path=base_env,
        sandbox_env_path=sandbox_env,
        include_sandbox=True,
    )

    assert (Path(base_env)).exists()
    assert (Path(sandbox_env)).exists()
    assert os.getenv("NOTION_TOKEN") == "base-token"
    assert os.getenv("NOTION_TASKS_DATABASE_ID") == "sandbox-db"
    assert os.getenv("NOTION_INBOX_PAGE_ID") == "sandbox-inbox"


@pytest.mark.unit
def test_initialize_runtime_environment_validates_required_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base_env = tmp_path / ".env"
    base_env.write_text("GOOGLE_API_KEY=test-google\n", encoding="utf-8")

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("NOTION_TOKEN", raising=False)

    with pytest.raises(EnvironmentError, match="NOTION_TOKEN"):
        initialize_runtime_environment(
            base_env_path=base_env,
            sandbox_env_path=tmp_path / ".env.sandbox",
            required_keys=("GOOGLE_API_KEY", "NOTION_TOKEN"),
        )
