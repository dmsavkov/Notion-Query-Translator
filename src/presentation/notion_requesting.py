"""Fetch Notion page properties and markdown body using the requests library.

No notion_client dependency — all calls go through plain HTTP requests
authenticated via the NOTION_TOKEN environment variable.
"""

import os
from typing import Any, Dict, Optional

import requests
from requests.exceptions import HTTPError


NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = os.getenv("NOTION_VERSION", "2022-06-28")
REQUEST_TIMEOUT = 30


def _notion_headers() -> Dict[str, str]:
    token = os.environ.get("NOTION_TOKEN", "")
    if not token:
        raise EnvironmentError("Missing NOTION_TOKEN environment variable")
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
    }


def _request_notion_json(
    url: str, *, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    response = requests.get(
        url, headers=_notion_headers(), params=params, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Expected Notion API to return a JSON object")
    return payload


def fetch_page_properties(page_id: str) -> Dict[str, Any]:
    """Fetch the raw Notion page object (properties, metadata)."""
    return _request_notion_json(f"{NOTION_API_BASE}/pages/{page_id}")


def fetch_page_markdown(
    page_id: str, *, include_transcript: bool = False
) -> str:
    """Fetch page content via the markdown endpoint.

    Handles truncated blocks by recursively fetching unknown_block_ids,
    and gracefully skips permission-denied blocks (404s).
    """
    markdown_parts: list[str] = []
    pending_ids = [page_id]
    seen_ids: set[str] = set()

    while pending_ids:
        current_id = pending_ids.pop(0)
        if current_id in seen_ids:
            continue
        seen_ids.add(current_id)

        params = {"include_transcript": True} if include_transcript else None
        try:
            payload = _request_notion_json(
                f"{NOTION_API_BASE}/pages/{current_id}/markdown", params=params
            )
        except HTTPError as e:
            # Skip inaccessible blocks but re-raise root page errors
            if current_id == page_id or e.response.status_code != 404:
                raise
            continue

        markdown = payload.get("markdown")
        if isinstance(markdown, str) and markdown.strip():
            markdown_parts.append(markdown.strip())

        for unknown_block_id in payload.get("unknown_block_ids", []):
            if (
                isinstance(unknown_block_id, str)
                and unknown_block_id.strip()
                and unknown_block_id not in seen_ids
            ):
                pending_ids.append(unknown_block_id.strip())

    return "\n\n".join(markdown_parts).strip()
