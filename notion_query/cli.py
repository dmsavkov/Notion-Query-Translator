import asyncio
from typing import Any, Dict

import typer

app = typer.Typer(help="CLI for running the Notion agent pipeline.")


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
    """Run the Notion agent pipeline for a single user prompt.

    This command wraps the async pipeline entrypoint and constructs the
    required tasks payload from `user_prompt`.
    """
    tasks: Dict[str, Dict[str, Any]] = {
        "user_request": {
            "query": user_prompt,
            "think": think,
        }
    }

    from run_pipeline import main

    try:
        result = asyncio.run(main(eval_tasks=tasks))
    except Exception as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    
    # Extract the single task result
    task_result = result.get("user_request", {})
    
    # Display results
    passed = task_result.get("passed", False)
    output = task_result.get("output", "")
    final_code = task_result.get("final_code", "")
    error = task_result.get("error")
    
    if error:
        typer.secho(f"❌ Error: {error}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    
    status_color = typer.colors.GREEN if passed else typer.colors.YELLOW
    status_text = "✓ PASS" if passed else "✗ FAIL"
    typer.secho(status_text, fg=status_color, bold=True)

    if passed:
        typer.echo("\n" + "="*60)
        typer.echo("Execution Output:")
        typer.echo("="*60)
        typer.echo(output or "(no output)")
    
    if final_code:
        typer.echo("\n" + "="*60)
        typer.echo("Generated Code:")
        typer.echo("="*60)
        typer.echo(final_code)


if __name__ == "__main__":
    app()
