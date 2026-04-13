"""Interactive shell helpers for the Notion CLI.

Contains prompt animation and shell command dispatching logic so the CLI
entrypoint can stay focused on orchestration.
"""

from __future__ import annotations

import asyncio
import shlex
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app


@dataclass
class ShellSettings:
    """Session-scoped shell flags."""

    think: bool = False
    max_rendered_relevant_page_ids: int = 5


@dataclass
class ShellDispatchResult:
    """Result of parsing one line of shell input."""

    action: Literal["run", "clear", "exit", "continue"]
    prompt: str = ""
    message: str = ""


def create_prompt_session() -> PromptSession[str]:
    return PromptSession()


def format_shell_status(settings: ShellSettings) -> str:
    mode = "ON" if settings.think else "OFF"
    return (
        "Session config: "
        f"think={mode}, max_rendered_relevant_page_ids={settings.max_rendered_relevant_page_ids}. "
        "Commands: /config [--think|--no-think] [--max-rendered-relevant-page-ids N], /clear, /exit"
    )


def _handle_config_command(args: list[str], settings: ShellSettings) -> ShellDispatchResult:
    if not args:
        return ShellDispatchResult(action="continue", message=f"[dim]{format_shell_status(settings)}[/dim]")

    allowed = {"--think", "--no-think", "--max-rendered-relevant-page-ids"}
    invalid = [arg for arg in args if arg not in allowed and not arg.isdigit()]
    if invalid:
        return ShellDispatchResult(
            action="continue",
            message="[yellow]Usage:[/yellow] /config [--think|--no-think] [--max-rendered-relevant-page-ids N]",
        )

    has_think = "--think" in args
    has_no_think = "--no-think" in args
    if has_think and has_no_think:
        return ShellDispatchResult(
            action="continue",
            message="[yellow]Choose only one:[/yellow] --think or --no-think",
        )

    if has_think:
        settings.think = True
    if has_no_think:
        settings.think = False

    if "--max-rendered-relevant-page-ids" in args:
        value_index = args.index("--max-rendered-relevant-page-ids") + 1
        if value_index >= len(args) or not args[value_index].isdigit():
            return ShellDispatchResult(
                action="continue",
                message="[yellow]Usage:[/yellow] /config [--think|--no-think] [--max-rendered-relevant-page-ids N]",
            )
        settings.max_rendered_relevant_page_ids = max(int(args[value_index]), 0)

    return ShellDispatchResult(action="continue", message=f"[dim]{format_shell_status(settings)}[/dim]")


def dispatch_shell_input(user_input: str, settings: ShellSettings) -> ShellDispatchResult:
    """Parse one user input line into a shell action."""
    text = str(user_input or "").strip()
    if not text:
        return ShellDispatchResult(action="continue")

    if not text.startswith("/"):
        return ShellDispatchResult(action="run", prompt=text)

    try:
        tokens = shlex.split(text)
    except ValueError:
        return ShellDispatchResult(action="continue", message="[yellow]Invalid command syntax.[/yellow]")

    if not tokens:
        return ShellDispatchResult(action="continue")

    command = tokens[0].lower()
    args = tokens[1:]

    if command in {"/exit", "/quit"}:
        return ShellDispatchResult(action="exit")

    if command == "/clear":
        return ShellDispatchResult(action="clear")

    if command == "/config":
        return _handle_config_command(args, settings)

    return ShellDispatchResult(
        action="continue",
        message="[yellow]Unknown command.[/yellow] Use /config, /clear, /exit",
    )


async def get_animated_input(
    session: PromptSession[str],
    *,
    waiting_label: str = "Waiting for prompt",
) -> str:
    """Render a slow animated waiting placeholder until the user types."""
    dots = 1

    def dynamic_prompt() -> str:
        try:
            if get_app().current_buffer.text:
                return "\n"
        except Exception:
            pass
        return f"{waiting_label}{'.' * dots}{' ' * (3 - dots)} \n"

    async def animate() -> None:
        nonlocal dots
        while True:
            await asyncio.sleep(0.4)
            dots = (dots % 3) + 1
            try:
                get_app().invalidate()
            except Exception:
                pass

    animator = asyncio.create_task(animate())
    try:
        return await session.prompt_async(dynamic_prompt)
    finally:
        animator.cancel()
        with suppress(asyncio.CancelledError):
            await animator