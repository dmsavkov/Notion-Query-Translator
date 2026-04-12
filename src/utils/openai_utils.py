"""OpenAI async client factory and request-scoped lifecycle (contextvars + session)."""

from __future__ import annotations

import contextlib
import contextvars
import os
from typing import Any, AsyncGenerator

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


_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

_current_openai_client: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "current_openai_client",
    default=None,
)


def get_openai_client() -> Any:
    client = _current_openai_client.get()
    if client is None:
        raise RuntimeError(
            "No AsyncOpenAI client found in context. "
            "Use `async with openai_client_session():` or run inside `run_with_lifecycle`."
        )
    return client


@contextlib.asynccontextmanager
async def openai_client_session(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> AsyncGenerator[Any, None]:
    key = (os.getenv("GOOGLE_API_KEY") or "") if api_key is None else api_key
    url = _DEFAULT_BASE_URL if base_url is None else base_url
    client = create_async_openai_client(api_key=key, base_url=url, **kwargs)
    token = _current_openai_client.set(client)
    try:
        yield client
    finally:
        _current_openai_client.reset(token)
        await client.close()
