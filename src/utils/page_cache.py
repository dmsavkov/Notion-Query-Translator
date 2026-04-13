"""In-memory async page cache for overlapping Notion fetch latency.

This cache is meant to be injected via RunnableConfig.configurable and must NOT
be stored in LangGraph state (it is not serializable due to asyncio Tasks).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..presentation.notion_requesting import (
    fetch_page_markdown,
    fetch_page_properties,
    resolve_block_to_page_id,
)


@dataclass(frozen=True)
class CachedPage:
    page_id: str
    properties: Dict[str, Any]
    markdown: str
    error: Optional[str] = None
    source_id: Optional[str] = None


def _normalize_id(value: str) -> str:
    return str(value or "").strip()


class AsyncPageCache:
    """Task registry with dedupe and optional refresh."""

    def __init__(self) -> None:
        self._tasks: Dict[str, asyncio.Task[CachedPage]] = {}

    def enqueue(self, page_or_block_id: str, *, force_refresh: bool = False) -> None:
        raw_id = _normalize_id(page_or_block_id)
        if not raw_id:
            return

        existing = self._tasks.get(raw_id)
        if existing is not None and not force_refresh:
            return

        if existing is not None and force_refresh and not existing.done():
            existing.cancel()

        self._tasks[raw_id] = asyncio.create_task(self._fetch_one(raw_id))

    async def _fetch_one(self, raw_id: str) -> CachedPage:
        try:
            # 1. Unpack both the ID and the potentially pre-fetched payload
            page_id, props = await asyncio.to_thread(self._resolve_and_fetch_properties, raw_id)
            
            if not page_id:
                return CachedPage(
                    page_id="",
                    properties={},
                    markdown="",
                    error="Unable to resolve ID to a page.",
                    source_id=raw_id,
                )

            # 2. Only make the network call if the resolver didn't already get the props
            if props is None:
                props = await asyncio.to_thread(fetch_page_properties, page_id)
                
            md = await asyncio.to_thread(fetch_page_markdown, page_id)
            return CachedPage(page_id=page_id, properties=props, markdown=md, source_id=raw_id)
            
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return CachedPage(
                page_id="", properties={}, markdown="",
                error=str(exc), source_id=raw_id,
            )

    @staticmethod
    def _resolve_and_fetch_properties(raw_id: str) -> tuple[str | None, dict | None]:
        """Returns (page_id, properties_payload). Payload is None if un-fetched."""
        
        # Fast path: treat as page ID.
        try:
            payload = fetch_page_properties(raw_id)
            page_id = payload.get("id") if isinstance(payload, dict) else None
            if isinstance(page_id, str) and page_id.strip():
                # We paid the network cost, don't throw the payload away!
                return page_id.strip(), payload 
        except Exception:
            pass

        # Fallback path: treat as block ID and normalize to parent page.
        try:
            parent_page_id = resolve_block_to_page_id(raw_id)
            # We only resolved the parent ID, we haven't fetched its properties yet.
            return parent_page_id, None 
        except Exception:
            return None, None

    async def gather_all(self) -> Dict[str, CachedPage]:
        if not self._tasks:
            return {}

        keys = list(self._tasks.keys())
        tasks = [self._tasks[k] for k in keys]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out: Dict[str, CachedPage] = {}
        for raw_id, result in zip(keys, results):
            if isinstance(result, CachedPage):
                out[raw_id] = result
            else:
                out[raw_id] = CachedPage(
                    page_id="",
                    properties={},
                    markdown="",
                    error=str(result),
                    source_id=raw_id,
                )
        return out

    def cancel_all(self) -> None:
        for task in self._tasks.values():
            if not task.done():
                task.cancel()

