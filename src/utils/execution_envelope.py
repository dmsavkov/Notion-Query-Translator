"""Shared helpers for the generated-code execution envelope.

These utilities keep the generated script's stdout envelope parsing,
page-id normalization, and render-cap logic out of the LangGraph node
implementation so the node stays focused on control flow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import json

from .execution_utils import ExecutionResult


DEFAULT_RENDER_CAP = 5


def normalize_page_ids(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple, set)):
        return []

    normalized: List[str] = []
    seen: set[str] = set()
    for raw_id in value:
        page_id = str(raw_id or "").strip()
        if not page_id or page_id in seen:
            continue
        seen.add(page_id)
        normalized.append(page_id)
    return normalized


def parse_execution_envelope(execution_output: str) -> Optional[Dict[str, Any]]:
    lines = [line.strip() for line in str(execution_output or "").splitlines() if line.strip()]
    if not lines:
        return None

    try:
        payload = json.loads(lines[-1])
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    message_to_user = payload.get("message_to_user")
    relevant_page_ids = payload.get("relevant_page_ids")
    if not isinstance(message_to_user, str) or not isinstance(relevant_page_ids, list):
        return None

    execution_status = str(payload.get("execution_status") or "").strip().lower()
    if execution_status not in {"success", "error"}:
        execution_status = ""

    return {
        "execution_status": execution_status,
        "message_to_user": message_to_user.strip(),
        "relevant_page_ids": normalize_page_ids(relevant_page_ids),
    }


def resolve_render_cap(pipeline_params: Any) -> int:
    raw_cap = getattr(pipeline_params, "max_rendered_relevant_page_ids", DEFAULT_RENDER_CAP)
    try:
        return max(int(raw_cap), 0)
    except (TypeError, ValueError):
        return DEFAULT_RENDER_CAP


def merge_display_page_ids(*, relevant_page_ids: List[str], mutated_page_ids: List[str], render_cap: int) -> List[str]:
    capped_relevant = relevant_page_ids[:render_cap]
    return dedupe_page_ids([*capped_relevant, *mutated_page_ids])


def dedupe_page_ids(ids: List[str]) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for raw_id in ids:
        page_id = str(raw_id or "").strip()
        if not page_id or page_id in seen:
            continue
        seen.add(page_id)
        normalized.append(page_id)
    return normalized


def apply_execution_envelope(
    *,
    result: ExecutionResult,
    update_data: Dict[str, Any],
    mutated_page_ids: List[str],
    pipeline_params: Any,
) -> List[str]:
    normalized_mutated = normalize_page_ids(mutated_page_ids)
    envelope = parse_execution_envelope(result.stdout)

    execution_status = "success" if result.passed else "error"
    message_to_user = ""
    relevant_page_ids: List[str] = []
    affected_notion_ids = normalized_mutated

    if envelope is not None:
        relevant_page_ids = envelope["relevant_page_ids"]
        message_to_user = envelope["message_to_user"]
        execution_status = envelope["execution_status"] or execution_status
        affected_notion_ids = merge_display_page_ids(
            relevant_page_ids=relevant_page_ids,
            mutated_page_ids=normalized_mutated,
            render_cap=resolve_render_cap(pipeline_params),
        )
        if message_to_user:
            update_data["execution_output"] = message_to_user
    elif result.passed:
        message_to_user = str(result.stdout or "").strip()

    update_data["execution_status"] = execution_status
    update_data["message_to_user"] = message_to_user
    update_data["relevant_page_ids"] = relevant_page_ids
    update_data["affected_notion_ids"] = affected_notion_ids

    return dedupe_page_ids([*relevant_page_ids, *normalized_mutated])