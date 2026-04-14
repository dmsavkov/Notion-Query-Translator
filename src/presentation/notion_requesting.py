"""Fetch Notion page properties and markdown body using the requests library.

No notion_client dependency — all calls go through plain HTTP requests
authenticated via the NOTION_TOKEN environment variable.
"""

import os
import re
from typing import Any, Dict, Optional

import requests
from requests.exceptions import HTTPError


NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = os.getenv("NOTION_VERSION", "2022-06-28")
REQUEST_TIMEOUT = 30


def normalize_title_for_match(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text)


def extract_result_title(result: Dict[str, Any]) -> str:
    props = result.get("properties")
    if isinstance(props, dict):
        for prop_value in props.values():
            if not isinstance(prop_value, dict) or prop_value.get("type") != "title":
                continue
            title_items = prop_value.get("title")
            if not isinstance(title_items, list):
                continue
            text = "".join(str(item.get("plain_text", "")) for item in title_items if isinstance(item, dict)).strip()
            if text:
                return text

    if isinstance(result.get("title"), str) and str(result.get("title")).strip():
        return str(result.get("title")).strip()

    return "Untitled"


def pick_exact_or_first_match_id(title: str, matches: list[Dict[str, Any]]) -> str:
    normalized_title = normalize_title_for_match(title)
    first_valid_id = ""
    for item in matches:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            continue
        if not first_valid_id:
            first_valid_id = item_id
        if normalize_title_for_match(extract_result_title(item)) == normalized_title:
            return item_id
    return first_valid_id


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


def fetch_block(block_id: str) -> Dict[str, Any]:
    """Fetch the raw Notion block object (content, parent pointers)."""
    return _request_notion_json(f"{NOTION_API_BASE}/blocks/{block_id}")


def resolve_block_to_page_id(block_id: str, *, max_depth: int = 20) -> Optional[str]:
    """Resolve a block ID to its owning page ID when possible.

    Walks `parent.block_id` chains up to `max_depth` to find a `parent.page_id`.
    Returns None when no page parent can be determined.
    """
    current_id = str(block_id or "").strip()
    if not current_id:
        return None

    visited: set[str] = set()
    for _ in range(max_depth):
        if current_id in visited:
            return None
        visited.add(current_id)

        data = fetch_block(current_id)
        parent = data.get("parent") if isinstance(data, dict) else None
        if not isinstance(parent, dict):
            return None

        parent_type = parent.get("type")
        if parent_type == "page_id":
            page_id = parent.get("page_id")
            return str(page_id).strip() if isinstance(page_id, str) and page_id.strip() else None
        if parent_type == "block_id":
            next_id = parent.get("block_id")
            if not isinstance(next_id, str) or not next_id.strip():
                return None
            current_id = next_id.strip()
            continue

        # database_id, workspace, or unknown parent type
        return None

    return None


def search_pages_by_title(title: str, limit: int = 10) -> list[Dict[str, Any]]:
    """Search for Notion pages matching the given title."""
    payload = {
        "query": title,
        "filter": {"value": "page", "property": "object"},
        "page_size": limit,
    }
    response = requests.post(
        f"{NOTION_API_BASE}/search",
        headers=_notion_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Expected Notion API to return a JSON object")
    
    results = payload.get("results", [])
    return [r for r in results if not r.get("archived")]



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
