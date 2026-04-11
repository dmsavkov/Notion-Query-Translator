from unittest.mock import patch

import pytest
import typer

from notion_query import cli as cli_module


@pytest.mark.unit
def test_run_without_prompt_enters_interactive_shell():
    with patch.object(cli_module, "initialize_runtime_environment") as mock_init:
        with patch.object(cli_module, "_setup_log_routing"):
            with patch.object(cli_module, "interactive_shell") as mock_shell:
                cli_module.run(user_prompt=None, think=True)

    mock_init.assert_called_once_with(required_keys=("NOTION_TOKEN", "GOOGLE_API_KEY"))
    mock_shell.assert_called_once_with(initial_think=True)


@pytest.mark.unit
def test_run_with_prompt_uses_single_shot_iteration():
    with patch.object(cli_module, "initialize_runtime_environment") as mock_init:
        with patch.object(cli_module, "_setup_log_routing"):
            with patch.object(cli_module, "_run_iteration", return_value={}) as mock_iteration:
                with patch.object(cli_module.console, "print"):
                    cli_module.run(user_prompt="Get my tasks", think=False)

    mock_init.assert_called_once_with(required_keys=("NOTION_TOKEN", "GOOGLE_API_KEY"))
    mock_iteration.assert_called_once_with(user_prompt="Get my tasks", think=False)


@pytest.mark.unit
def test_run_with_prompt_exits_nonzero_when_iteration_reports_error():
    with patch.object(cli_module, "initialize_runtime_environment") as mock_init:
        with patch.object(cli_module, "_setup_log_routing"):
            with patch.object(cli_module, "_run_iteration", return_value={"error": "boom"}):
                with patch.object(cli_module.console, "print"):
                    with pytest.raises(typer.Exit) as exc:
                        cli_module.run(user_prompt="Get my tasks", think=False)

    mock_init.assert_called_once_with(required_keys=("NOTION_TOKEN", "GOOGLE_API_KEY"))
    assert exc.value.exit_code == 1


@pytest.mark.unit
def test_run_exits_nonzero_when_environment_is_invalid():
    with patch.object(cli_module, "initialize_runtime_environment", side_effect=EnvironmentError("missing key")):
        with patch.object(cli_module.console, "print"):
            with pytest.raises(typer.Exit) as exc:
                cli_module.run(user_prompt="Get my tasks", think=False)

    assert exc.value.exit_code == 1
