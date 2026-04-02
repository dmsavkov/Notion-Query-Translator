"""Shared OpenAI client helpers."""

from typing import Any

import openai
from langsmith.wrappers import wrap_openai


def create_async_openai_client(*, base_url: str, api_key: str, **kwargs: Any) -> Any:
    client_kwargs = {
        "base_url": base_url,
        "api_key": api_key,
        "max_retries": 25,
    }
    client_kwargs.update(kwargs)
    return wrap_openai(openai.AsyncOpenAI(**client_kwargs))