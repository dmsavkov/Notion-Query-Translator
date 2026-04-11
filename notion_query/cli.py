"""CLI entry point for the Notion Query Translator agent.

Provides a clean, rich terminal UI with:
- A splash screen introducing the application.
- A dynamic spinner that reflects real-time graph node transitions.
- All backend stdout routed to ``logs/logs.log`` to keep the UI pristine.
- Post-execution rendering of affected Notion pages via the presentation layer.
"""

import asyncio
import logging
import os
from contextlib import redirect_stdout

import typer
from rich.console import Console
from rich.panel import Panel

from src.adapters.cli_factory import build_app_config_from_cli
from src.core.lifecycle import run_with_lifecycle
from src.models.config import AppConfig
from src.models.schema import CliParams, build_cli_eval_tasks
from src.presentation import ui_bridge
from src.presentation.cli_helpers import cli_disambiguator
from src.presentation.viewer import print_completed_state

app = typer.Typer(help="Notion Query Translator — natural language to Notion API.")
console = Console(stderr=True)  # UI output goes to stderr to stay clean

# ---------------------------------------------------------------------------
# Status map: backend node name → rich spinner text
# ---------------------------------------------------------------------------
STATUS_MAP = {
    "initializing": "[dim]Waking up the agent...[/dim]",
    "precheck_general": "[dim]Running general precheck...[/dim]",
    "precheck_security": "[dim]Scanning request for safety...[/dim]",
    "precheck_join": "[dim]Collecting precheck results...[/dim]",
    "awaiting_disambiguation": "[bold cyan]Waiting for your selection...[/bold cyan]",
    "malovolent_request": "[bold red]Request flagged — preparing rejection...[/bold red]",
    "retrieve": "[bold cyan]Retrieving Notion API documentation...[/bold cyan]",
    "plan": "[bold cyan]Planning the implementation...[/bold cyan]",
    "codegen": "[bold yellow]Writing Python script for Notion API...[/bold yellow]",
    "execute_local": "[bold magenta]Executing code locally...[/bold magenta]",
    "execute_sandbox": "[bold magenta]Booting E2B sandbox and running code...[/bold magenta]",
    "egress_security": "[bold cyan]Checking output for token leaks...[/bold cyan]",
    "reflect": "[bold red]Execution issue detected — rewriting (attempt {trial})...[/bold red]",
    "cleanup_sandbox": "[dim]Cleaning up sandbox...[/dim]",
    "completed": "[bold green]✓ Task complete.[/bold green]",
    "error": "[bold red]An error occurred.[/bold red]",
}

# ---------------------------------------------------------------------------
# Splash screen
# ---------------------------------------------------------------------------
SPLASH = """[bold cyan]
 ╔══════════════════════════════════════════════════════╗
 ║          Notion Query Translator  v0.1              ║
 ║  Translate natural language → Notion API actions    ║
 ╚══════════════════════════════════════════════════════╝[/bold cyan]

 [dim]Powered by LangGraph · E2B Sandbox · Rich CLI[/dim]
"""


def _setup_log_routing() -> None:
    """Route Python logging and captured stdout to logs/logs.log."""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/logs.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


async def _run_with_spinner(
    tasks: dict,
    app_config: AppConfig,
) -> dict:
    """Execute the pipeline while polling ui_bridge to animate the spinner."""

    with console.status(STATUS_MAP["initializing"], spinner="dots") as status:
        async def _update_spinner_loop() -> None:
            last_seen_node = None
            while True:
                current = ui_bridge.current_node
                if current != last_seen_node:
                    text = STATUS_MAP.get(current, f"[dim]Processing: {current}...[/dim]")
                    # Inject dynamic trial count for the reflect node
                    if "{trial}" in text:
                        text = text.format(trial=ui_bridge.trial_num)
                    status.update(text)
                    last_seen_node = current
                await asyncio.sleep(0.1)

        poller_task = asyncio.create_task(_update_spinner_loop())

        try:
            # Route all backend stdout into the log file
            log_file = open("logs/logs.log", "a", encoding="utf-8")
            try:
                with redirect_stdout(log_file):
                    result = await run_with_lifecycle(
                        tasks=tasks, app_config=app_config
                    )
            finally:
                log_file.close()
        finally:
            poller_task.cancel()

    return result


@app.command()
def run(
    user_prompt: str,
    think: bool = typer.Option(
        False,
        "--think",
        "-t",
        help="Enable reflection (not 'Minimal' mode)",
    ),
) -> None:
    """Run the Notion agent pipeline for a single user prompt."""

    _setup_log_routing()

    # Splash screen
    console.print(SPLASH)
    console.print(f"[bold]Prompt:[/bold] {user_prompt}\n")

    # Reset bridge state
    ui_bridge.current_node = "initializing"
    ui_bridge.trial_num = 0
    ui_bridge.disambiguator = cli_disambiguator

    try:
        cli_params = CliParams(user_prompt=user_prompt, think=think)
        app_config = build_app_config_from_cli(cli_params=cli_params)
        eval_tasks = build_cli_eval_tasks(cli_params)
        result = asyncio.run(
            _run_with_spinner(tasks=eval_tasks, app_config=app_config)
        )
    except Exception as exc:
        console.print(
            Panel(
                f"[bold red]Fatal error:[/bold red] {exc}",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    # Extract the single task result and render it
    task_result = result.get("user_request", {})
    error = task_result.get("error")

    # Render using the presentation layer (this goes to real stdout)
    output_console = Console()  # fresh console on stdout
    with output_console.capture():
        pass  # ensure stdout is usable
    print_completed_state(task_result)

    if error:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
