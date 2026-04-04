import asyncio

import typer
from rich.console import Console

from src.adapters.cli_factory import build_app_config_from_cli
from src.adapters.cli_presenter import format_lifecycle_result
from src.core.lifecycle import run_with_lifecycle
from src.models.schema import CliParams, build_cli_eval_tasks

app = typer.Typer(help="CLI for running the Notion agent pipeline.")
console = Console()


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

    try:
        cli_params = CliParams(user_prompt=user_prompt, think=think)
        app_config = build_app_config_from_cli(cli_params=cli_params)
        eval_tasks = build_cli_eval_tasks(cli_params)
        result = asyncio.run(run_with_lifecycle(tasks=eval_tasks, app_config=app_config))
    except Exception as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    task_result = result.get("user_request", {})
    error = task_result.get("error")

    console.print(format_lifecycle_result(result))

    if error:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
