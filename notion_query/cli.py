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
        help="Deprecated: reflection mode toggle is currently ignored.",
    ),
) -> None:
    """Run the Notion agent pipeline for a single user prompt.

    This command wraps the async pipeline entrypoint and constructs the
    required tasks payload from `user_prompt`.
    """
    tasks: Dict[str, Dict[str, Any]] = {
        "user_request": {
            "query": user_prompt,
        }
    }

    # Intentionally unused for now; reserved for future reflection controls.
    _ = think

    from run_pipeline import main

    asyncio.run(main(eval_tasks=tasks))


if __name__ == "__main__":
    app()
