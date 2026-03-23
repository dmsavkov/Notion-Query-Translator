import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

import pyperclip
import requests
from dotenv import load_dotenv
from langsmith import Client

from .all_functionality import async_chat_wrapper
from .config import _MODEL_MAP


GROUP_PROMPT_PATH = Path("instructions/helpful-prompts/chatbot_group_report_prompt.md")
ANAL_PROMPT_PATH = Path("instructions/helpful-prompts/chatbot_anal_prompt.md")
NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
JUDGE_PRIMARY_MODEL_ALIAS = "gemini_flash_latest"
JUDGE_FALLBACK_MODEL_ALIAS = "gemini-3.1-flash-lite-preview"
MAX_NOTION_RICH_TEXT = 1800
DEFAULT_DATASET_NAME = "Dataset v4."


class RunRecord(TypedDict, total=False):
    experiment: str
    task_id: str
    thread_id: str
    run_id: str
    scores: Dict[str, Any]
    final_code: str
    retrieval_context: str
    comments: Dict[str, Any]
    outputs: Dict[str, Any]


@dataclass
class HumanConfig:
    """User-facing toggles controlling which analysis sections are exported.

    Section switches:
    - include_code: export generated code snapshots.
    - include_code_execution: export execution-failure summaries.
    - include_code_statements: export code statement judge outputs.
    - include_rag: export retrieval-context snapshots.
    - include_rag_statements: export RAG statement judge outputs.
    - include_plans: export request plan snapshots.
    - include_all_in_one: export one fully consolidated run snapshot.
    - include_code_mismatches: export cross-score mismatch cases.

    Limits:
    - max_examples_per_field=None means no cap.
    - max_*_chars=None means no truncation.

    Statement filtering:
    - statement_status_filter="both" keeps right + wrong statement rows.
    - statement_status_filter="wrong" keeps only wrong rows.
    - statement_status_filter="right" keeps only right rows.

    Judging:
    - judging_enabled: whether to run LLM judges on section outputs.
    """

    include_code: bool = True
    include_code_execution: bool = True
    include_code_statements: bool = True
    include_rag: bool = True
    include_rag_statements: bool = True
    include_plans: bool = True
    include_all_in_one: bool = True
    include_code_mismatches: bool = True
    judging_enabled: bool = True
    statement_status_filter: Literal["both", "wrong", "right"] = "both"
    max_examples_per_field: Optional[int] = None
    max_code_chars: Optional[int] = None
    max_rag_chars: Optional[int] = None
    max_plan_chars: Optional[int] = None
    llm_max_concurrency: int = 5


@dataclass
class SectionPayload:
    section_name: str
    enabled: bool
    output: Dict[str, Any]
    prompt: str
    judge_output: str


def _ensure_model_aliases() -> None:
    _MODEL_MAP.setdefault(JUDGE_PRIMARY_MODEL_ALIAS, "gemini-flash-latest")
    _MODEL_MAP.setdefault(JUDGE_FALLBACK_MODEL_ALIAS, "gemini-3.1-flash-lite-preview")


def _safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _chunk_text(value: str, max_chars: int = MAX_NOTION_RICH_TEXT) -> List[str]:
    if not value:
        return ["(empty)"]
    chunks: List[str] = []
    start = 0
    while start < len(value):
        chunks.append(value[start : start + max_chars])
        start += max_chars
    return chunks


def _parse_comment_payload(comment_value: Any) -> Any:
    if comment_value is None:
        return None
    if isinstance(comment_value, (dict, list)):
        return comment_value
    if not isinstance(comment_value, str):
        return comment_value

    text = comment_value.strip()
    if not text:
        return ""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_experiment_runs(
    experiment_prefix: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> List[RunRecord]:
    client = Client()
    records: List[RunRecord] = []

    projects = list(client.list_projects(reference_dataset_name=dataset_name))
    projects = [p for p in projects if str(getattr(p, "name", "") or "").startswith(experiment_prefix)]

    for project in projects:
        runs = list(client.list_runs(project_name=project.name, is_root=True))
        if not runs:
            continue

        run_ids = [str(r.id) for r in runs]
        feedback_by_run: Dict[str, Dict[str, Any]] = {rid: {} for rid in run_ids}
        for fb in client.list_feedback(run_ids=run_ids):
            rid = str(fb.run_id)
            if fb.score is not None:
                feedback_by_run[rid][fb.key] = fb.score
            if fb.comment:
                feedback_by_run[rid][f"{fb.key}_comment"] = fb.comment

        for run in runs:
            rid = str(run.id)
            feedback_data = feedback_by_run.get(rid, {})
            scores = {k: v for k, v in feedback_data.items() if not k.endswith("_comment")}

            outputs = run.outputs or {}
            pre_state = outputs.get("pre_computed_state") or {}
            task_id = (
                str(outputs.get("task_id") or "").strip()
                or str(pre_state.get("task_id") or "").strip()
            )

            comments: Dict[str, Any] = {}
            for key, value in feedback_data.items():
                if not key.endswith("_comment"):
                    continue
                comment_key = key.replace("_comment", "")
                comments[comment_key] = _parse_comment_payload(value)

            record: RunRecord = {
                "experiment": str(getattr(project, "name", "") or ""),
                "task_id": task_id,
                "thread_id": outputs.get("thread_id", ""),
                "run_id": rid,
                "scores": scores,
                "final_code": pre_state.get("final_code") or pre_state.get("generated_code") or "",
                "retrieval_context": pre_state.get("retrieval_context", ""),
                "comments": comments,
                "outputs": outputs,
            }
            records.append(record)

    return records


def _is_right_statement_status(status: str) -> bool:
    return status.strip().lower() in ("present", "pass", "true")


def _include_statement_status(status: str, status_filter: Literal["both", "wrong", "right"]) -> bool:
    is_right = _is_right_statement_status(status)
    if status_filter == "both":
        return True
    if status_filter == "right":
        return is_right
    return not is_right


def _statement_items(
    records: List[RunRecord],
    score_key: str,
    status_filter: Literal["both", "wrong", "right"],
    max_examples: Optional[int],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    for record in records:
        comments = record.get("comments", {}) or {}
        statement_items = comments.get(score_key)
        if not isinstance(statement_items, list):
            continue

        for item in statement_items:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", ""))
            if not _include_statement_status(status, status_filter):
                continue

            items.append(
                {
                    "task_id": record.get("task_id", ""),
                    "run_id": record.get("run_id", ""),
                    "thread_id": record.get("thread_id", ""),
                    "status": status,
                    "statement": item.get("statement", ""),
                    "reasoning": item.get("reasoning", ""),
                    "evidence": item.get("evidence", ""),
                }
            )

            if max_examples is not None and len(items) >= max_examples:
                return items

    return items


def _truncate_text(value: str, max_chars: Optional[int]) -> str:
    text = value or ""
    if max_chars is None:
        return text
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _extract_plan_from_record(record: RunRecord, max_chars: int) -> Dict[str, Any]:
    outputs = record.get("outputs", {}) or {}
    pre_state = outputs.get("pre_computed_state", {}) or {}

    candidates = {
        "outputs.plan": outputs.get("plan"),
        "outputs.request_plan": outputs.get("request_plan"),
        "pre_computed_state.plan": pre_state.get("plan"),
        "pre_computed_state.request_plan": pre_state.get("request_plan"),
    }

    for source, value in candidates.items():
        if isinstance(value, str) and value.strip():
            return {
                "source": source,
                "plan": _truncate_text(value.strip(), max_chars),
            }
        if isinstance(value, (dict, list)) and value:
            return {
                "source": source,
                "plan": _truncate_text(_safe_json(value), max_chars),
            }

    return {
        "source": "",
        "plan": "",
    }


def _record_snapshot(
    record: RunRecord,
    *,
    max_code_chars: int,
    max_rag_chars: int,
    max_plan_chars: int,
) -> Dict[str, Any]:
    return {
        "experiment": record.get("experiment", ""),
        "task_id": record.get("task_id", ""),
        "run_id": record.get("run_id", ""),
        "thread_id": record.get("thread_id", ""),
        "scores": record.get("scores", {}) or {},
        "code": _truncate_text(record.get("final_code", "") or "", max_code_chars),
        "rag": _truncate_text(record.get("retrieval_context", "") or "", max_rag_chars),
        "plan": _extract_plan_from_record(record, max_plan_chars),
        "comments": record.get("comments", {}) or {},
        "pre_computed_state": ((record.get("outputs", {}) or {}).get("pre_computed_state") or {}),
    }


def _build_code_output(records: List[RunRecord], config: HumanConfig) -> Dict[str, Any]:
    max_items = config.max_examples_per_field if config.max_examples_per_field is not None else len(records)
    examples = [
        {
            "task_id": record.get("task_id", ""),
            "run_id": record.get("run_id", ""),
            "thread_id": record.get("thread_id", ""),
            "scores": record.get("scores", {}) or {},
            "code": _truncate_text(record.get("final_code", "") or "", config.max_code_chars),
        }
        for record in records[:max_items]
    ]

    return {
        "total_runs": len(records),
        "examples": examples,
    }


def _build_rag_output(records: List[RunRecord], config: HumanConfig) -> Dict[str, Any]:
    max_items = config.max_examples_per_field if config.max_examples_per_field is not None else len(records)
    examples = [
        {
            "task_id": record.get("task_id", ""),
            "run_id": record.get("run_id", ""),
            "thread_id": record.get("thread_id", ""),
            "rag": _truncate_text(record.get("retrieval_context", "") or "", config.max_rag_chars),
        }
        for record in records[:max_items]
    ]

    return {
        "total_runs": len(records),
        "examples": examples,
    }


def _build_plans_output(records: List[RunRecord], config: HumanConfig) -> Dict[str, Any]:
    examples: List[Dict[str, Any]] = []
    for record in records:
        plan_info = _extract_plan_from_record(record, config.max_plan_chars)
        if not plan_info["plan"]:
            continue

        examples.append(
            {
                "task_id": record.get("task_id", ""),
                "run_id": record.get("run_id", ""),
                "thread_id": record.get("thread_id", ""),
                "source": plan_info["source"],
                "plan": plan_info["plan"],
            }
        )
        if config.max_examples_per_field is not None and len(examples) >= config.max_examples_per_field:
            break

    return {
        "total_runs": len(records),
        "plans_found": len(examples),
        "examples": examples,
    }


def _build_all_in_one_output(records: List[RunRecord], config: HumanConfig) -> Dict[str, Any]:
    if not records:
        return {
            "total_runs": 0,
            "record": {},
        }

    record = records[0]
    return {
        "total_runs": len(records),
        "record": _record_snapshot(
            record,
            max_code_chars=config.max_code_chars,
            max_rag_chars=config.max_rag_chars,
            max_plan_chars=config.max_plan_chars,
        ),
    }


def _build_code_execution_output(records: List[RunRecord], max_examples: Optional[int]) -> Dict[str, Any]:
    failed = [
        {
            "task_id": r.get("task_id", ""),
            "run_id": r.get("run_id", ""),
            "thread_id": r.get("thread_id", ""),
            "code_execution_score": (r.get("scores", {}) or {}).get("code_execution_score"),
            "code_preview": (r.get("final_code", "") or "")[:500],
        }
        for r in records
        if float((r.get("scores", {}) or {}).get("code_execution_score", 0.0) or 0.0) < 1.0
    ]

    return {
        "total_runs": len(records),
        "failed_runs": len(failed),
        "examples": failed[:max_examples] if max_examples is not None else failed,
    }


def _build_code_statements_output(records: List[RunRecord], config: HumanConfig) -> Dict[str, Any]:
    statements = _statement_items(
        records,
        "code_statements_score",
        config.statement_status_filter,
        config.max_examples_per_field,
    )

    return {
        "total_runs": len(records),
        "status_filter": config.statement_status_filter,
        "statement_count": len(statements),
        "examples": statements,
    }


def _build_rag_statements_output(records: List[RunRecord], config: HumanConfig) -> Dict[str, Any]:
    statements = _statement_items(
        records,
        "rag_statements_score",
        config.statement_status_filter,
        config.max_examples_per_field,
    )

    return {
        "total_runs": len(records),
        "status_filter": config.statement_status_filter,
        "statement_count": len(statements),
        "examples": statements,
    }


def _build_code_mismatches_output(records: List[RunRecord], config: HumanConfig) -> Dict[str, Any]:
    cases_exec_pass_stmt_not_perfect: List[Dict[str, Any]] = []
    cases_exec_fail_stmt_perfect: List[Dict[str, Any]] = []

    for record in records:
        scores = record.get("scores", {}) or {}
        exec_score = float(scores.get("code_execution_score", 0.0) or 0.0)
        stmt_score = float(scores.get("code_statements_score", 0.0) or 0.0)

        row = {
            "task_id": record.get("task_id", ""),
            "run_id": record.get("run_id", ""),
            "thread_id": record.get("thread_id", ""),
            "code_execution_score": exec_score,
            "code_statements_score": stmt_score,
            "code_preview": _truncate_text(record.get("final_code", "") or "", config.max_code_chars),
        }

        if exec_score == 1.0 and stmt_score < 1.0:
            cases_exec_pass_stmt_not_perfect.append(row)
        elif exec_score != 1.0 and stmt_score == 1.0:
            cases_exec_fail_stmt_perfect.append(row)

    if config.max_examples_per_field is not None:
        cases_exec_pass_stmt_not_perfect = cases_exec_pass_stmt_not_perfect[: config.max_examples_per_field]
        cases_exec_fail_stmt_perfect = cases_exec_fail_stmt_perfect[: config.max_examples_per_field]

    return {
        "total_runs": len(records),
        "exec_pass_stmt_not_perfect_count": len(cases_exec_pass_stmt_not_perfect),
        "exec_fail_stmt_perfect_count": len(cases_exec_fail_stmt_perfect),
        "exec_pass_stmt_not_perfect_examples": cases_exec_pass_stmt_not_perfect,
        "exec_fail_stmt_perfect_examples": cases_exec_fail_stmt_perfect,
    }


def _read_prompt_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_group_report_prompt(artifacts: str, template: Optional[str] = None) -> str:
    base = template if template is not None else _read_prompt_template(GROUP_PROMPT_PATH)
    marker = "[[``````]]"
    if marker in base:
        return base.replace(marker, artifacts)
    return f"{base}\n\n[ARTIFACTS]:\n{artifacts}"


def build_architecture_analysis_prompt(
    synthesized_error_data: str,
    template: Optional[str] = None,
) -> str:
    base = template if template is not None else _read_prompt_template(ANAL_PROMPT_PATH)
    marker = "[[INSERT_ERROR_DATA_HERE]]"
    if marker in base:
        return base.replace(marker, synthesized_error_data)
    return f"{base}\n\nSynthesized Error Data:\n{synthesized_error_data}"


async def _judge_with_fallback(prompt: str) -> str:
    try:
        response = await async_chat_wrapper(
            [{"role": "user", "content": prompt}],
            json_output=False,
            max_tokens=1600,
            temperature=0.2,
            model_size=JUDGE_PRIMARY_MODEL_ALIAS,
        )
        return str(response)
    except Exception:
        response = await async_chat_wrapper(
            [{"role": "user", "content": prompt}],
            json_output=False,
            max_tokens=1600,
            temperature=0.2,
            model_size=JUDGE_FALLBACK_MODEL_ALIAS,
        )
        return str(response)


async def _run_group_judges(
    payloads: List[SectionPayload],
    *,
    max_concurrency: int,
) -> List[SectionPayload]:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _run_one(payload: SectionPayload) -> SectionPayload:
        if not payload.enabled:
            return payload
        if not payload.output:
            payload.judge_output = "No output data available for this section."
            return payload

        async with semaphore:
            payload.judge_output = await _judge_with_fallback(payload.prompt)
            return payload

    return await asyncio.gather(*[_run_one(p) for p in payloads])


def _build_section_payloads(records: List[RunRecord], config: HumanConfig) -> List[SectionPayload]:
    builders: List[tuple[str, bool, Callable[[List[RunRecord], HumanConfig], Dict[str, Any]]]] = [
        ("code", config.include_code, lambda rs, cfg: _build_code_output(rs, cfg)),
        (
            "code_execution",
            config.include_code_execution,
            lambda rs, cfg: _build_code_execution_output(rs, cfg.max_examples_per_field),
        ),
        (
            "code_statements",
            config.include_code_statements,
            lambda rs, cfg: _build_code_statements_output(rs, cfg),
        ),
        ("rag", config.include_rag, lambda rs, cfg: _build_rag_output(rs, cfg)),
        (
            "rag_statements",
            config.include_rag_statements,
            lambda rs, cfg: _build_rag_statements_output(rs, cfg),
        ),
        ("plans", config.include_plans, lambda rs, cfg: _build_plans_output(rs, cfg)),
        ("all_in_one", config.include_all_in_one, lambda rs, cfg: _build_all_in_one_output(rs, cfg)),
        (
            "code_mismatches",
            config.include_code_mismatches,
            lambda rs, cfg: _build_code_mismatches_output(rs, cfg),
        ),
    ]

    payloads: List[SectionPayload] = []
    for section_name, enabled, builder in builders:
        output = builder(records, config) if enabled else {}
        artifacts = _safe_json(output)
        prompt = build_group_report_prompt(artifacts)
        payloads.append(
            SectionPayload(
                section_name=section_name,
                enabled=enabled,
                output=output,
                prompt=prompt,
                judge_output="",
            )
        )

    return payloads


def _notion_headers() -> Dict[str, str]:
    token = os.getenv("NOTION_TOKEN_GLOBAL", "").strip()
    if not token:
        raise EnvironmentError("Missing NOTION_TOKEN_GLOBAL environment variable")

    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }


def _notion_rich_text(text: str) -> List[Dict[str, Any]]:
    return [{"type": "text", "text": {"content": text}}]


def _append_children(block_id: str, children: List[Dict[str, Any]]) -> None:
    headers = _notion_headers()
    for i in range(0, len(children), 100):
        chunk = children[i : i + 100]
        response = requests.patch(
            f"{NOTION_API_BASE}/blocks/{block_id}/children",
            headers=headers,
            json={"children": chunk},
            timeout=30,
        )
        response.raise_for_status()


def create_tracking_page(title: str) -> str:
    tracking_page_id = os.getenv("NOTION_TRACKING_PAGE_ID", "").strip()
    if not tracking_page_id:
        raise EnvironmentError("Missing NOTION_TRACKING_PAGE_ID environment variable")

    headers = _notion_headers()
    payload = {
        "parent": {"type": "page_id", "page_id": tracking_page_id},
        "properties": {
            "title": {
                "title": [
                    {
                        "type": "text",
                        "text": {"content": title},
                    }
                ]
            }
        },
    }

    response = requests.post(f"{NOTION_API_BASE}/pages", headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["id"]


def _build_notion_children(payloads: List[SectionPayload], exp_name: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": _notion_rich_text(f"Experiment: {exp_name}")
            },
        },
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": _notion_rich_text(
                    "This page stores LLM grouped diagnostics and raw field outputs for manual inspection."
                )
            },
        },
    ]

    for payload in payloads:
        if not payload.enabled:
            continue

        blocks.append(
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": _notion_rich_text(payload.section_name)},
            }
        )

        judge_text = payload.judge_output or "No judge output produced."
        for chunk in _chunk_text(judge_text):
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": _notion_rich_text(chunk)},
                }
            )

    raw_children: List[Dict[str, Any]] = []
    for payload in payloads:
        if not payload.enabled:
            continue

        section_text = _safe_json(payload.output)
        item_children = [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": _notion_rich_text(chunk)},
            }
            for chunk in _chunk_text(section_text)
        ]
        raw_children.append(
            {
                "object": "block",
                "type": "toggle",
                "toggle": {
                    "rich_text": _notion_rich_text(payload.section_name),
                    "children": item_children,
                },
            }
        )

    blocks.append(
        {
            "object": "block",
            "type": "toggle",
            "toggle": {
                "rich_text": _notion_rich_text("Raw error analysis outputs"),
                "children": raw_children,
            },
        }
    )

    return blocks


def _synthesized_text_for_clipboard(payloads: List[SectionPayload]) -> str:
    lines: List[str] = []
    for payload in payloads:
        if not payload.enabled:
            continue
        lines.append(f"## {payload.section_name}")
        lines.append(payload.judge_output or "No judge output produced.")
        lines.append("")
    return "\n".join(lines).strip()


def _copy_to_clipboard(text: str) -> None:
    pyperclip.copy(text)


async def run_error_analysis(
    exp_name: str,
    config: Optional[HumanConfig] = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> Dict[str, Any]:
    load_dotenv()
    _ensure_model_aliases()

    cfg = config or HumanConfig()
    records = load_experiment_runs(experiment_prefix=exp_name, dataset_name=dataset_name)
    payloads = _build_section_payloads(records, cfg)
    
    if cfg.judging_enabled:
        judged_payloads = await _run_group_judges(payloads, max_concurrency=cfg.llm_max_concurrency)
    else:
        judged_payloads = payloads

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    page_title = f"ERROR_ANALYSIS_{exp_name}_{timestamp}"
    notion_page_id = create_tracking_page(page_title)

    children = _build_notion_children(judged_payloads, exp_name)
    _append_children(notion_page_id, children)

    grouped_only = _synthesized_text_for_clipboard(judged_payloads)
    final_prompt = build_architecture_analysis_prompt(grouped_only)
    _copy_to_clipboard(final_prompt)

    return {
        "page_id": notion_page_id,
        "page_title": page_title,
        "record_count": len(records),
        "sections": [
            {
                "section_name": p.section_name,
                "enabled": p.enabled,
                "output": p.output,
                "judge_output": p.judge_output,
            }
            for p in judged_payloads
        ],
        "clipboard_prompt": final_prompt,
    }


def main(exp_name: str, config: Optional[HumanConfig] = None, dataset_name: str = DEFAULT_DATASET_NAME) -> Dict[str, Any]:
    return asyncio.run(run_error_analysis(exp_name=exp_name, config=config, dataset_name=dataset_name))

