"""Rich terminal viewer for completed pipeline state.

Orchestrates data fetching (notion_requesting), sanitization, and
rendering via the ``rich`` library.  Called by the CLI after the
LangGraph state machine hits __end__.
"""

import json
import sys
from typing import Any, Dict, List

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from .notion_requesting import fetch_page_markdown, fetch_page_properties
from .sanitization import (
    IGNORE_PROPS,
    flatten_notion_properties,
    sanitize_notion_markdown,
)


# Theme configuration
THEME = {
    "header_style": "cyan",
    "header_rule": "bold blue",
    "table_header": "bold magenta",
    "panel_style": "dim",
}


def _extract_page_title(flat_props: Dict[str, Any]) -> str:
    """Try common naming patterns to find the page title."""
    for key in ("Name", "Title", "title", "name"):
        if key in flat_props and flat_props[key]:
            return str(flat_props[key])
    page_id = flat_props.get("page_id", "unknown")
    return f"Page {str(page_id)[:8]}"


def _render_single_page(console: Console, page_id: str) -> None:
    """Fetch, flatten, and render a single Notion page."""
    try:
        raw_props = fetch_page_properties(page_id)
        raw_md = fetch_page_markdown(page_id)
    except Exception as exc:
        console.print(
            Panel(
                f"[red]Failed to fetch page [bold]{page_id}[/bold]:[/red]\n{exc}",
                border_style="red",
            )
        )
        return

    flat_props = flatten_notion_properties(raw_props)
    clean_md = sanitize_notion_markdown(raw_md)
    page_title = _extract_page_title(flat_props)

    # Header
    console.print(
        Rule(
            f"[{THEME['header_rule']}]{page_title}[/{THEME['header_rule']}]",
            style=THEME["header_style"],
        )
    )

    # Properties table
    table = Table(
        title="Page Properties",
        show_header=True,
        header_style=THEME["table_header"],
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("Property", style="cyan", no_wrap=False)
    table.add_column("Value", style="green")

    for key, value in flat_props.items():
        if not value or key.lower() in ("name", "title") or key in IGNORE_PROPS:
            continue
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value, ensure_ascii=False, default=str)[:200]
        else:
            value_str = str(value)[:200]
        table.add_row(key, value_str)

    console.print(table)

    # Markdown body
    if clean_md:
        md = Markdown(clean_md)
        console.print(
            Panel(
                md,
                title="[bold green]Document Body[/bold green]",
                border_style=THEME["panel_style"],
            )
        )
    else:
        console.print("[dim]No content available[/dim]")


def print_completed_state(final_state: Dict[str, Any]) -> None:
    """Inspect the pipeline's terminal state and render results.

    - On success: fetches each affected Notion page and pretty-prints
      its properties + markdown body.
    - On failure / security-blocked: shows an error panel.
    """
    console = Console()
    terminal_status = final_state.get("terminal_status", "pending")
    affected_ids: List[str] = final_state.get("affected_notion_ids", [])

    # --- Failure paths ---
    if terminal_status == "security_blocked":
        console.print(
            Panel(
                "[bold red]⛔ Request blocked by security policy.[/bold red]\n\n"
                + str(final_state.get("execution_output", "")),
                title="Security Block",
                border_style="red",
            )
        )
        return

    if terminal_status in ("execution_failed", "max_retries_exceeded"):
        error_detail = final_state.get("execution_output", "")
        feedback = final_state.get("feedback", "")
        body = f"[red]{error_detail}[/red]"
        if feedback:
            body += f"\n\n[dim]Feedback:[/dim] {feedback}"
        console.print(
            Panel(
                body,
                title=f"[bold red]Execution Failed ({terminal_status})[/bold red]",
                border_style="red",
            )
        )
        return

    # --- Success path ---
    if not affected_ids:
        # Show raw execution output when no specific pages were affected
        output = final_state.get("execution_output", "(no output)")
        console.print(
            Panel(
                str(output),
                title="[bold yellow]Execution Output (no specific pages affected)[/bold yellow]",
                border_style="yellow",
            )
        )
        return

    console.print(
        f"\n[bold cyan]📄 {len(affected_ids)} page(s) affected[/bold cyan]\n"
    )
    for page_id in affected_ids:
        _render_single_page(console, page_id)
        console.print()  # spacing
