"""Sanitize and flatten Notion API responses for clean terminal rendering.

Handles:
- Property flattening (rich_text, select, multi_select, dates, relations, etc.)
- Markdown cleanup (HTML table conversion, Notion-specific tag handling)
- Structure compaction (strip empty keys)
"""

import json
import re
from typing import Any, Dict


# Properties to omit from the user-facing table
IGNORE_PROPS = {
    "page_object",
    "page_parent",
    "page_public_url",
    "page_last_edited_time",
    "page_archived",
    "page_locked",
    "page_icon",
    "page_cover",
}


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------

def _compact_structure(value: Any) -> Any:
    """Recursively strip None / empty values from nested dicts and lists."""
    if isinstance(value, dict):
        compacted: Dict[str, Any] = {}
        for key, item in value.items():
            simplified = _compact_structure(item)
            if simplified in (None, "", [], {}):
                continue
            compacted[key] = simplified
        return compacted

    if isinstance(value, list):
        compacted_list = []
        for item in value:
            simplified = _compact_structure(item)
            if simplified in (None, "", [], {}):
                continue
            compacted_list.append(simplified)
        return compacted_list

    return value


def _simplify_user(user: Any) -> Any:
    if not isinstance(user, dict):
        return user
    simplified = {
        "id": user.get("id"),
        "name": user.get("name"),
        "object": user.get("object"),
        "type": user.get("type"),
        "avatar_url": user.get("avatar_url"),
    }
    person = user.get("person")
    if isinstance(person, dict):
        simplified["email"] = person.get("email")
    bot = user.get("bot")
    if isinstance(bot, dict) and bot:
        simplified["bot"] = _compact_structure(bot)
    return _compact_structure(simplified)


def _simplify_file(file_obj: Any) -> Any:
    if not isinstance(file_obj, dict):
        return file_obj
    simplified = {"name": file_obj.get("name"), "type": file_obj.get("type")}
    file_data = file_obj.get("file")
    if isinstance(file_data, dict):
        simplified["url"] = file_data.get("url")
        simplified["expiry_time"] = file_data.get("expiry_time")
    external_data = file_obj.get("external")
    if isinstance(external_data, dict):
        simplified["url"] = external_data.get("url")
    return _compact_structure(simplified)


def _join_rich_text(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    pieces: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        plain_text = item.get("plain_text")
        if isinstance(plain_text, str):
            pieces.append(plain_text)
            continue
        text = item.get("text")
        if isinstance(text, dict):
            content = text.get("content")
            if isinstance(content, str):
                pieces.append(content)
                continue
        mention = item.get("mention")
        if isinstance(mention, dict):
            mention_text = (
                mention.get("plain_text")
                or mention.get("name")
                or mention.get("id")
            )
            if mention_text is not None:
                pieces.append(str(mention_text))
    return "".join(pieces).strip()


def _flatten_formula_value(formula: Any) -> Any:
    if not isinstance(formula, dict):
        return formula
    formula_type = formula.get("type")
    if isinstance(formula_type, str):
        return {
            "type": formula_type,
            "value": _compact_structure(formula.get(formula_type)),
        }
    return _compact_structure(formula)


def _flatten_rollup_value(rollup: Any) -> Any:
    if not isinstance(rollup, dict):
        return rollup
    rollup_type = rollup.get("type")
    if rollup_type == "array" and isinstance(rollup.get("array"), list):
        value = [_flatten_property_value(item) for item in rollup.get("array", [])]
    elif isinstance(rollup_type, str) and rollup_type in rollup:
        value = _compact_structure(rollup.get(rollup_type))
    else:
        value = _compact_structure(rollup.get("array"))
    return _compact_structure(
        {"type": rollup_type, "function": rollup.get("function"), "value": value}
    )


def _simplify_verification(verification: Any) -> Any:
    if not isinstance(verification, dict):
        return verification
    return _compact_structure(
        {
            "state": verification.get("state"),
            "date": verification.get("date"),
            "verified_by": _simplify_user(verification.get("verified_by")),
        }
    )


def _flatten_property_value(property_value: Any) -> Any:
    """Convert a single Notion property value into a simple Python value."""
    if not isinstance(property_value, dict):
        return property_value

    pt = property_value.get("type")

    if pt == "title":
        return _join_rich_text(property_value.get("title"))
    if pt == "rich_text":
        return _join_rich_text(property_value.get("rich_text"))
    if pt in {"select", "status"}:
        option = property_value.get(pt)
        if isinstance(option, dict):
            return option.get("name") or option.get("id")
        return option
    if pt == "multi_select":
        options = property_value.get("multi_select")
        if isinstance(options, list):
            return [
                o.get("name") or o.get("id")
                for o in options
                if isinstance(o, dict) and (o.get("name") or o.get("id"))
            ]
        return options
    if pt == "checkbox":
        return bool(property_value.get("checkbox"))
    if pt == "number":
        return property_value.get("number")
    if pt in {"url", "email", "phone_number", "created_time", "last_edited_time"}:
        return property_value.get(pt)
    if pt == "date":
        return _compact_structure(property_value.get("date"))
    if pt in {"created_by", "last_edited_by"}:
        return _simplify_user(property_value.get(pt))
    if pt == "people":
        people = property_value.get("people")
        if isinstance(people, list):
            return [_simplify_user(p) for p in people if isinstance(p, dict)]
        return people
    if pt == "relation":
        relations = property_value.get("relation")
        if isinstance(relations, list):
            return [
                r.get("id")
                for r in relations
                if isinstance(r, dict) and r.get("id")
            ]
        return relations
    if pt == "files":
        files = property_value.get("files")
        if isinstance(files, list):
            return [_simplify_file(f) for f in files if isinstance(f, dict)]
        return files
    if pt == "formula":
        return _flatten_formula_value(property_value.get("formula"))
    if pt == "rollup":
        return _flatten_rollup_value(property_value.get("rollup"))
    if pt == "unique_id":
        uid = property_value.get("unique_id")
        if isinstance(uid, dict):
            prefix = uid.get("prefix")
            number = uid.get("number")
            if prefix not in (None, ""):
                return f"{prefix}-{number}"
            return number
        return uid
    if pt == "verification":
        return _simplify_verification(property_value.get("verification"))

    # Fallback for untyped properties
    if "title" in property_value:
        return _join_rich_text(property_value.get("title"))
    if "rich_text" in property_value:
        return _join_rich_text(property_value.get("rich_text"))

    return _compact_structure(property_value)


# ---------------------------------------------------------------------------
# Property flattening (top-level page object → simple dict)
# ---------------------------------------------------------------------------

def flatten_notion_properties(raw_props: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the Notion page object into a context-friendly mapping."""
    if not isinstance(raw_props, dict):
        return {}

    flattened: Dict[str, Any] = {
        "page_object": raw_props.get("object"),
        "page_id": raw_props.get("id"),
        "page_url": raw_props.get("url"),
        "page_public_url": raw_props.get("public_url"),
        "page_created_time": raw_props.get("created_time"),
        "page_last_edited_time": raw_props.get("last_edited_time"),
        "page_in_trash": raw_props.get("in_trash"),
        "page_archived": raw_props.get("is_archived"),
        "page_locked": raw_props.get("is_locked"),
    }

    if raw_props.get("parent") is not None:
        flattened["page_parent"] = _compact_structure(raw_props.get("parent"))
    if raw_props.get("icon") is not None:
        flattened["page_icon"] = _compact_structure(raw_props.get("icon"))
    if raw_props.get("cover") is not None:
        flattened["page_cover"] = _compact_structure(raw_props.get("cover"))

    properties = raw_props.get("properties")
    if isinstance(properties, dict):
        for prop_name, prop_value in properties.items():
            flattened[str(prop_name)] = _flatten_property_value(prop_value)

    return {k: v for k, v in flattened.items() if v not in (None, "", [], {})}


# ---------------------------------------------------------------------------
# Markdown sanitization
# ---------------------------------------------------------------------------

def _extract_html_table(html_text: str) -> tuple[list[list[str]], bool]:
    """Extract table data from HTML tags using simple regex (avoids BS4 dep)."""
    table_match = re.search(r"<table[^>]*>(.*?)</table>", html_text, re.DOTALL | re.IGNORECASE)
    if not table_match:
        return ([], False)

    rows: list[list[str]] = []
    for tr_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", table_match.group(1), re.DOTALL | re.IGNORECASE):
        cells = [
            re.sub(r"<[^>]+>", "", cell).strip()
            for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr_match.group(1), re.DOTALL | re.IGNORECASE)
        ]
        if cells:
            rows.append(cells)

    return (rows, len(rows) > 0)


def _markdown_table_from_rows(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    lines: list[str] = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            sep = "| " + " | ".join("-" * max(3, len(c)) for c in row) + " |"
            lines.append(sep)
    return "\n".join(lines)


def sanitize_notion_markdown(raw_md: Any) -> str:
    """Normalize markdown returned by the Notion markdown endpoint.

    Conversions:
    - <summary>Text</summary> → **► Text**
    - <callout icon="💡">Text</callout> → > 💡 Text
    - <page url="X">Y</page> → [Y](X)
    - <database url="X">Title</database> → [Title](X)
    - <mention-page url="X" .../> → [Referenced](X)
    - HTML tables → Markdown tables
    - Remaining structural HTML stripped
    """
    if raw_md is None:
        return ""

    markdown = raw_md.get("markdown") if isinstance(raw_md, dict) else raw_md
    if markdown is None:
        return ""
    if not isinstance(markdown, str):
        markdown = str(markdown)

    # Normalize line endings
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
    markdown = markdown.replace("\u00a0", " ")

    # HTML tables → markdown tables
    rows, has_table = _extract_html_table(markdown)
    if has_table:
        md_table = _markdown_table_from_rows(rows)
        markdown = re.sub(
            r"<table[^>]*>.*?</table>", md_table, markdown,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # <summary> → bold toggle
    def _replace_summary(m: re.Match) -> str:
        return f"{m.group(1)}**► {m.group(2).strip()}**"

    markdown = re.sub(
        r"^(\s*)<summary[^>]*>([^<]*)</summary>",
        _replace_summary, markdown, flags=re.IGNORECASE | re.MULTILINE,
    )

    # <callout icon="X"> → blockquote
    def _replace_callout(m: re.Match) -> str:
        icon_match = re.search(r'icon="([^"]*)"', m.group(2))
        icon = icon_match.group(1) if icon_match else "💬"
        return f"\n{m.group(1)}> {icon} {m.group(3).strip()}\n"

    markdown = re.sub(
        r"^(\s*)<callout\s+([^>]*)>([^<]*)</callout>",
        _replace_callout, markdown, flags=re.IGNORECASE | re.MULTILINE,
    )

    # <page url="X">Y</page> → link
    def _replace_page(m: re.Match) -> str:
        return f"\n{m.group(1)}[{m.group(3).strip()}]({m.group(2)})\n"

    markdown = re.sub(
        r'^(\s*)<page\s+[^>]*url="([^"]*)"[^>]*>([^<]*)</page>',
        _replace_page, markdown, flags=re.IGNORECASE | re.MULTILINE,
    )

    # <database url="X">Title</database> → link
    def _replace_database(m: re.Match) -> str:
        text = m.group(3).strip() or "Database"
        return f"\n{m.group(1)}[{text}]({m.group(2)})\n"

    markdown = re.sub(
        r'^(\s*)<database\s+[^>]*url="([^"]*)"[^>]*>([^<]*)</database>',
        _replace_database, markdown, flags=re.IGNORECASE | re.MULTILINE,
    )

    # <mention-page url="X" .../> → link
    def _replace_mention_page(m: re.Match) -> str:
        url_match = re.search(r'url="([^"]*)"', m.group(2))
        url = url_match.group(1) if url_match else ""
        return f"\n{m.group(1)}[Referenced]({url})\n"

    markdown = re.sub(
        r"^(\s*)<mention-page\s+([^>]*)/>",
        _replace_mention_page, markdown, flags=re.IGNORECASE | re.MULTILINE,
    )

    # Strip remaining structural HTML
    markdown = re.sub(
        r"<(details|table|tr|td|colgroup|col)[^>]*>", "", markdown, flags=re.IGNORECASE
    )
    markdown = re.sub(
        r"</(details|table|tr|td|colgroup|col)>", "", markdown, flags=re.IGNORECASE
    )
    markdown = re.sub(r"<empty-block\s*/?>", "", markdown, flags=re.IGNORECASE)

    # Clean up whitespace
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()
