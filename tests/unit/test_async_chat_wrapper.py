import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.all_functionality import async_chat_wrapper


def _mock_response(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ]
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_chat_wrapper_passes_temperature_and_adds_concision_prompt():
    messages = [{"role": "user", "content": "Say hello"}]
    create_mock = AsyncMock(return_value=_mock_response("hello"))

    with patch("src.all_functionality._async_client.chat.completions.create", create_mock):
        result = await async_chat_wrapper(
            messages=messages,
            max_tokens=321,
            temperature=0.42,
            json_output=False,
            model_size="gemma4",
        )

    assert result == "hello"
    call_kwargs = create_mock.await_args.kwargs
    assert call_kwargs["temperature"] == 0.42
    assert call_kwargs["model"] == "gemma-3-4b-it"
    assert len(call_kwargs["messages"]) == 2
    assert "UNDER 321 words" in call_kwargs["messages"][1]["content"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_chat_wrapper_json_mode_non_gemini_adds_json_instruction():
    messages = [{"role": "user", "content": "Return JSON"}]
    create_mock = AsyncMock(return_value=_mock_response('{"ok": true}'))

    with patch("src.all_functionality._async_client.chat.completions.create", create_mock):
        result = await async_chat_wrapper(
            messages=messages,
            max_tokens=150,
            temperature=0.31,
            json_output=True,
            model_size="gemma4",
        )

    assert result == {"ok": True}
    call_kwargs = create_mock.await_args.kwargs
    assert call_kwargs["temperature"] == 0.31
    assert len(call_kwargs["messages"]) == 3
    assert "UNDER 150 words" in call_kwargs["messages"][1]["content"]
    assert call_kwargs["messages"][2]["content"] == "Please provide the output in JSON format."
