"""Pure-state singleton for decoupled UI updates during LangGraph streaming.

This module contains NO UI formatting and NO third-party imports.
It acts as an isolated memory space that the backend writes to
and the frontend polls from asynchronously.
"""

# Current graph node being executed. Updated by execute_single's astream loop.
current_node: str = "initializing"

# Current retry attempt number, updated when the reflect node fires.
trial_num: int = 0
