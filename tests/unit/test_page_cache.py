import asyncio
from unittest.mock import AsyncMock

import pytest

from src.utils.page_cache import AsyncPageCache, CachedPage


@pytest.mark.unit
@pytest.mark.asyncio
async def test_page_cache_dedup_enqueue_calls_fetch_once(monkeypatch: pytest.MonkeyPatch):
    cache = AsyncPageCache()
    fetch = AsyncMock(return_value=CachedPage(page_id="p1", properties={}, markdown=""))
    monkeypatch.setattr(cache, "_fetch_one", fetch)

    cache.enqueue("id-1")
    cache.enqueue("id-1")

    out = await cache.gather_all()
    assert "id-1" in out
    assert fetch.await_count == 1
    assert cache._tasks == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_page_cache_force_refresh_replaces_task(monkeypatch: pytest.MonkeyPatch):
    cache = AsyncPageCache()

    gate = asyncio.Event()

    async def _slow(_raw_id: str) -> CachedPage:
        await gate.wait()
        return CachedPage(page_id="p1", properties={}, markdown="")

    monkeypatch.setattr(cache, "_fetch_one", _slow)

    cache.enqueue("id-1")
    first_task = cache._tasks["id-1"]
    cache.enqueue("id-1", force_refresh=True)
    second_task = cache._tasks["id-1"]

    assert first_task is not second_task
    gate.set()
    out = await cache.gather_all()
    assert "id-1" in out
    assert cache._tasks == {}

