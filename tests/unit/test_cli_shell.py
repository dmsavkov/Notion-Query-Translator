import pytest

from src.presentation.cli_shell import ShellSettings, dispatch_shell_input


@pytest.mark.unit
def test_dispatch_shell_input_plain_prompt_runs():
    settings = ShellSettings(think=False)
    result = dispatch_shell_input("Find my tasks", settings)

    assert result.action == "run"
    assert result.prompt == "Find my tasks"


@pytest.mark.unit
def test_dispatch_shell_input_exit_aliases():
    settings = ShellSettings(think=False)

    assert dispatch_shell_input("/exit", settings).action == "exit"
    assert dispatch_shell_input("/quit", settings).action == "exit"


@pytest.mark.unit
def test_dispatch_shell_input_clear():
    settings = ShellSettings(think=False)
    result = dispatch_shell_input("/clear", settings)

    assert result.action == "clear"


@pytest.mark.unit
def test_dispatch_shell_input_config_toggle():
    settings = ShellSettings(think=False)

    res_on = dispatch_shell_input("/config --think", settings)
    assert res_on.action == "continue"
    assert settings.think is True

    res_off = dispatch_shell_input("/config --no-think", settings)
    assert res_off.action == "continue"
    assert settings.think is False


@pytest.mark.unit
def test_dispatch_shell_input_config_rejects_conflict():
    settings = ShellSettings(think=False)
    result = dispatch_shell_input("/config --think --no-think", settings)

    assert result.action == "continue"
    assert "Choose only one" in result.message


@pytest.mark.unit
def test_dispatch_shell_input_unknown_command():
    settings = ShellSettings(think=False)
    result = dispatch_shell_input("/wat", settings)

    assert result.action == "continue"
    assert "Unknown command" in result.message