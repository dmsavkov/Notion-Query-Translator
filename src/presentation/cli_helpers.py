"""CLI helpers for interactive prompts and complex terminal UI logic.
"""

from contextlib import contextmanager
import sys
from typing import Dict, List, Optional

import questionary


@contextmanager
def _interactive_stdio():
    """Temporarily restore real terminal streams for interactive prompts.

    The CLI redirects backend stdout to log files during graph execution.
    Questionary must use the original TTY streams so the prompt remains visible.
    """
    previous_stdin = sys.stdin
    previous_stdout = sys.stdout
    sys.stdin = sys.__stdin__
    sys.stdout = sys.__stdout__
    try:
        yield
    finally:
        sys.stdin = previous_stdin
        sys.stdout = previous_stdout

async def cli_disambiguator(title: str, options: List[Dict[str, str]]) -> Optional[str]:
    """Interactive questionary prompt for resource resolution."""
    if not options:
        return None

    # Build list of display choices
    choices = []
    for opt in options:
        name = str(opt.get("title") or "Untitled")
        option_id = str(opt.get("id") or "")
        url = str(opt.get("url") or "")
        short_url = url.split("/")[-1] if url else "no-link"
        short_id = option_id[:8] if option_id else "unknown"
        choices.append(
            questionary.Choice(
                title=f"{name} ({short_id}, {short_url})",
                value=option_id,
            )
        )
    
    choices.append(questionary.Choice(title="[None of above - Cancel]", value=None))

    with _interactive_stdio():
        answer = await questionary.select(
            f"Multiple pages found for '{title}'. Please select the correct one:",
            choices=choices,
            style=questionary.Style([
                ("qmark", "fg:#673ab7 bold"),
                ("question", "bold"),
                ("answer", "fg:#f44336 bold"),
                ("pointer", "fg:#673ab7 bold"),
                ("highlighted", "fg:#673ab7 bold"),
                ("selected", "fg:#cc9900"),
                ("separator", "fg:#cc5454"),
                ("instruction", ""),
                ("text", ""),
                ("disabled", "fg:#858585 italic"),
            ]),
        ).ask_async()

    return answer
