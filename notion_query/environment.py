"""Runtime environment bootstrap helpers.

Entry points should call these helpers before invoking backend logic.
"""

from collections.abc import Sequence
from pathlib import Path
import os

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_ENV_PATH = REPO_ROOT / ".env"
SANDBOX_ENV_PATH = REPO_ROOT / ".env.sandbox"


def load_runtime_environment(
    *,
    base_env_path: Path | None = None,
    sandbox_env_path: Path | None = None,
    include_sandbox: bool = True,
) -> tuple[Path, Path]:
    """Load .env and optional .env.sandbox with sandbox override semantics."""
    base_path = base_env_path or BASE_ENV_PATH
    layered_sandbox_path = sandbox_env_path or SANDBOX_ENV_PATH

    if base_path.exists():
        load_dotenv(dotenv_path=base_path, override=False)

    if include_sandbox and layered_sandbox_path.exists():
        # Sandbox IDs must override static defaults from the base environment.
        load_dotenv(dotenv_path=layered_sandbox_path, override=True)

    return base_path, layered_sandbox_path


def validate_required_environment(required_keys: Sequence[str]) -> None:
    """Raise if any required environment variable is missing/empty."""
    missing = [key for key in required_keys if not (os.getenv(key) or "").strip()]
    if missing:
        joined = ", ".join(missing)
        raise EnvironmentError(f"Missing required environment variables: {joined}")


def initialize_runtime_environment(
    *,
    required_keys: Sequence[str] = (),
    include_sandbox: bool = True,
    base_env_path: Path | None = None,
    sandbox_env_path: Path | None = None,
) -> tuple[Path, Path]:
    """Load runtime env files and validate required keys."""
    paths = load_runtime_environment(
        base_env_path=base_env_path,
        sandbox_env_path=sandbox_env_path,
        include_sandbox=include_sandbox,
    )
    if required_keys:
        validate_required_environment(required_keys)
    return paths


__all__ = [
    "BASE_ENV_PATH",
    "SANDBOX_ENV_PATH",
    "initialize_runtime_environment",
    "load_runtime_environment",
    "validate_required_environment",
]
