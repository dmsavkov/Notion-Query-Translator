"""Pure-state singleton for decoupled UI updates during LangGraph streaming.

This module contains NO UI formatting and NO third-party imports.
It acts as an isolated memory space that the backend writes to
and the frontend polls from asynchronously.
"""
from typing import Any

# Current graph node being executed. Updated by execute_single's astream loop.
current_node: str = "initializing"

# Current retry attempt number, updated when the reflect node fires.
trial_num: int = 0

# Interactive callback for resource disambiguation.
# Signature: async def (title: str, options: list[dict]) -> str
disambiguator: Any = None

# Sentinel returned by the CLI cancel choice.
DISAMBIGUATION_CANCELLED: str = "__cancel_disambiguation__"

# Optional per-task in-memory artifacts (NOT part of LangGraph state).
# Used to harvest background caches after lifecycle completion.
page_caches: dict[str, Any] = {}
