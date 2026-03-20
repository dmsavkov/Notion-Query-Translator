"""Context variants loaded from files or hardcoded fallbacks."""

from typing import Dict
from pathlib import Path


# Hardcoded context strings (kept intact as requested)
_HARDCODED_STRINGS = {
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
}


def _load_context_files() -> Dict[str, str]:
    """Load all context files from the data/context directory."""
    context_dir = Path(__file__).parent.parent / "data" / "context"
    contexts = {}
    
    if context_dir.exists():
        for file_path in context_dir.iterdir():
            if file_path.is_file():
                # Use stem (filename without extension) as key
                key = file_path.stem
                try:
                    contexts[key] = file_path.read_text(encoding="utf-8")
                except Exception as e:
                    print(f"Warning: Could not load context file {file_path}: {e}")
    
    return contexts

def add_combinations(base: Dict[str, str]) -> Dict[str, str]:
    """Add combination contexts based on existing keys."""
    combined = {}
    combinations = [
        ['database_schema_report_token_efficient', 'notion_api_comprehensive_3'],
        ['database_schema_report_token_efficient', 'notion_api_top25_20220628'],
        ['database_schema_report_token_efficient', 'notion_api_top25'],
    ]
    
    for combo in combinations:
        keys = combo
        if all(k in base for k in keys):
            combined_key = "__".join(keys)
            combined[combined_key] = "\n\n".join(base[k] for k in keys)
    
    return combined

# Load all contexts: hardcoded strings + all files from directory
_FILE_CONTEXTS = _load_context_files()
_HARDCODED_CONTEXTS = {**_HARDCODED_STRINGS, **_FILE_CONTEXTS}
HARDCODED_CONTEXTS: Dict[str, str] = {**_HARDCODED_CONTEXTS, **add_combinations(_HARDCODED_CONTEXTS)}

# Type alias for available context keys (dynamically generated)
ContextUsed = str


def get_hardcoded_context(context_used: ContextUsed) -> str:
    """Return the selected hardcoded context variant.
    
    Args:
        context_used: Context name to retrieve. Available options are dynamically
                     determined from hardcoded strings and files in data/context directory.
    
    Returns:
        The content of the requested context.
    
    Raises:
        ValueError: If context_used is not available.
    """
    if context_used == "dynamic":
        raise ValueError("dynamic context does not map to a hardcoded literal")
    if context_used not in HARDCODED_CONTEXTS:
        available = sorted(HARDCODED_CONTEXTS.keys())
        raise ValueError(
            f"Unknown context: '{context_used}'. Available contexts: {available}"
        )
    return HARDCODED_CONTEXTS[context_used]

if __name__ == "__main__":
    print("keys", HARDCODED_CONTEXTS.keys())
