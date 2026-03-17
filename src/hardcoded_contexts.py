"""Context variants loaded from files or hardcoded fallbacks."""

from typing import Dict, Literal
from pathlib import Path


ContextUsed = Literal["dynamic", "baseline", "detailed", "personal_efficient_comprehensive_3", "personal_efficient", "personal_comprehensive", "comprehensive_3"]


def _load_file(relative_path: str) -> str:
    """Load a file from the data/context directory."""
    file_path = Path(__file__).parent.parent / "data" / "context" / relative_path
    return file_path.read_text(encoding="utf-8")

# Load files at module initialization
_DATABASE_SCHEMA_TOKEN_EFFICIENT = _load_file("database_schema_report_token_efficient.txt")
_DATABASE_SCHEMA_COMPREHENSIVE = _load_file("database_schema_report_comprehensive.txt")
_NOTION_API_COMPREHENSIVE = _load_file("notion_api_comprehensive_3.md")

HARDCODED_CONTEXTS: Dict[ContextUsed, str] = {
    "baseline": (
        "Notion API quick reference:\n"
        "- Create page: POST /v1/pages with parent and properties.\n"
        "- Query database: POST /v1/databases/{database_id}/query with optional filter/sorts.\n"
        "- Update page: PATCH /v1/pages/{page_id} for properties and archive status.\n"
        "- Retrieve block children: GET /v1/blocks/{block_id}/children.\n"
        "- Append block children: PATCH /v1/blocks/{block_id}/children.\n"
        "- Search: POST /v1/search with filter for page/database object types.\n"
        "Use Notion-Version: 2022-06-28 and bearer token authorization."
    ),
    "detailed": (
        "Notion API detailed operational context:\n"
        "1) Authentication:\n"
        "   - Include Authorization: Bearer <NOTION_TOKEN>.\n"
        "   - Include Notion-Version: 2022-06-28.\n"
        "   - Use Content-Type: application/json for write requests.\n"
        "2) Data access patterns:\n"
        "   - Databases are queried via POST /v1/databases/{database_id}/query.\n"
        "   - Pages are created via POST /v1/pages with parent database_id/page_id.\n"
        "   - Page properties are updated via PATCH /v1/pages/{page_id}.\n"
        "3) Property conventions:\n"
        "   - title: {\"title\": [{\"text\": {\"content\": ...}}]}\n"
        "   - select/status: set by option name where available.\n"
        "   - relation: provide list of related page ids.\n"
        "   - date: provide start date in YYYY-MM-DD when needed.\n"
        "4) Blocks:\n"
        "   - Read content with GET /v1/blocks/{block_id}/children.\n"
        "   - Append checklist/paragraph blocks with PATCH /v1/blocks/{block_id}/children.\n"
        "5) Safety and reliability:\n"
        "   - Validate required identifiers before request execution.\n"
        "   - Prefer deterministic filters and explicit property names.\n"
        "   - Handle empty results and non-200 errors with clear diagnostics."
    ),
    "personal_efficient": _DATABASE_SCHEMA_TOKEN_EFFICIENT,
    "personal_comprehensive": _DATABASE_SCHEMA_COMPREHENSIVE,
    "comprehensive_3": _DATABASE_SCHEMA_COMPREHENSIVE,
    "personal_efficient_comprehensive_3": (
        _DATABASE_SCHEMA_TOKEN_EFFICIENT + "\n\n" +
        _NOTION_API_COMPREHENSIVE
    ),
}


def get_hardcoded_context(context_used: ContextUsed) -> str:
    """Return the selected hardcoded context variant."""
    if context_used == "dynamic":
        raise ValueError("dynamic context does not map to a hardcoded literal")
    return HARDCODED_CONTEXTS[context_used]
