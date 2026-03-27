import asyncio

import typer
from rich.console import Console

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
    """Run the Notion agent pipeline for a single user prompt.

    This command wraps the async pipeline entrypoint and passes
    structured CLI parameters to `main`.
    """
    from run_pipeline import CliParams, main

    try:
        cli_params = CliParams(user_prompt=user_prompt, think=think)
        result = asyncio.run(main(cli_params=cli_params, dev_mode=False))
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
        console.print(f"ERROR: {error}")
        raise typer.Exit(code=1)
    
    if passed:
        console.print("PASS")
        console.print("\n[bold cyan]Execution Output:[/bold cyan]")
        console.print(output or "(no output)")
    else:
        console.print("FAIL")
        console.print("\n[bold yellow]Output:[/bold yellow]")
        console.print(output or "(no output)")
    
    if final_code:
        console.print("\n[bold cyan]Generated Code:[/bold cyan]")
        console.print(final_code)


if __name__ == "__main__":
    app()
