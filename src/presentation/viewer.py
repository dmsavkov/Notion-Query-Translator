"""Rich terminal viewer for completed pipeline state.

Orchestrates data fetching (notion_requesting), sanitization, and
rendering via the ``rich`` library.  Called by the CLI after the
LangGraph state machine hits __end__.
"""

import ast
import json
import sys
from typing import Any, Dict, List, Optional

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


def _extract_page_payload_candidates(execution_output: str) -> List[str]:
    """Return possible literal payload slices from stdout text."""
    lines = [line.rstrip() for line in str(execution_output or "").splitlines()]
    candidates: List[str] = []
    seen: set[str] = set()

    for index, line in enumerate(lines):
        stripped = line.lstrip()
        literal_start: Optional[int] = None

        if stripped.startswith(("{", "[")):
            literal_start = len(line) - len(stripped)
        else:
            for marker in ("{", "["):
                marker_index = line.find(marker)
                if marker_index != -1:
                    literal_start = marker_index
                    break

        if literal_start is None:
            continue

        candidate = "\n".join([line[literal_start:], *lines[index + 1 :]]).strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    return candidates


def _try_parse_page_payload(execution_output: str) -> Any:
    """Parse stdout as JSON or a Python literal if it looks like page output."""
    for candidate in _extract_page_payload_candidates(execution_output):
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except Exception:
                continue
    return None


def _collect_page_ids(value: Any) -> List[str]:
    """Recursively collect Notion page IDs from parsed stdout payloads."""
    page_ids: List[str] = []

    def _walk(item: Any) -> None:
        if isinstance(item, dict):
            if item.get("object") == "page":
                page_id = item.get("id")
                if isinstance(page_id, str) and page_id.strip():
                    page_ids.append(page_id.strip())
            for nested in item.values():
                _walk(nested)
        elif isinstance(item, list):
            for nested in item:
                _walk(nested)

    _walk(value)

    deduped: List[str] = []
    seen: set[str] = set()
    for page_id in page_ids:
        if page_id in seen:
            continue
        seen.add(page_id)
        deduped.append(page_id)
    return deduped


def _prepare_completed_state(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """Promote page-shaped stdout into the normal page rendering flow."""
    if str(final_state.get("terminal_status") or "") != "success":
        return final_state

    if final_state.get("affected_notion_ids"):
        return final_state

    parsed_output = _try_parse_page_payload(str(final_state.get("execution_output") or ""))
    page_ids = _collect_page_ids(parsed_output) if parsed_output is not None else []
    if not page_ids:
        return final_state

    prepared_state = dict(final_state)
    prepared_state["affected_notion_ids"] = page_ids
    prepared_state["execution_output"] = ""
    return prepared_state


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
    final_state = _prepare_completed_state(final_state)
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

    if terminal_status == "resource_not_found":
        error_detail = final_state.get("execution_output", "Resource search failed.")
        console.print(
            Panel(
                f"[bold yellow]🔍 {error_detail}[/bold yellow]\n\nPlease check the spelling of the page in Notion and try again.",
                title="Search Failed",
                border_style="yellow",
            )
        )
        return

    if terminal_status == "ambiguity_unresolved":
        error_detail = final_state.get("execution_output", "Selection was cancelled.")
        console.print(
            Panel(
                f"[bold cyan]ℹ️ {error_detail}[/bold cyan]\n\nThe request was stopped because the correct page could not be identified.",
                title="Resolution Cancelled",
                border_style="cyan",
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
