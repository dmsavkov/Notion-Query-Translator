"""Telemetry code injection for tracking affected Notion page IDs.

Provides the "sandwich" wrapper that prepends an HTTP interceptor
and appends an ID dump to the LLM-generated code, so that after
execution we can deterministically read back which pages were touched.
"""

import textwrap
import json

# The file path where affected IDs are written inside the execution environment.
AFFECTED_IDS_PATH = "/tmp/affected_ids.json"

# Local execution uses a workspace-relative path instead.
LOCAL_AFFECTED_IDS_PATH = "data/tmp_affected_ids.json"

_TELEMETRY_HEADER = textwrap.dedent("""\
    import requests as __sys_requests
    import json as __sys_json
    import re as __sys_re
    import os as __sys_os
    import sys as __sys_sys

    __system_affected_ids = set()
    __current_request = __sys_requests.Session.request
    __original_request = getattr(__current_request, "__telemetry_original__", __current_request)
    __uuid_pattern = __sys_re.compile(
        r'[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}'
    )

    def __telemetry_request(self, method, url, *args, **kwargs):
        if method.upper() in ("GET", "PATCH", "POST", "DELETE"):
            if "api.notion.com/v1/pages/" in url or "api.notion.com/v1/blocks/" in url:
                try:
                    endpoint = "/pages/" if "/pages/" in url else "/blocks/"
                    page_id = url.split(endpoint)[1].split("?")[0].split("/")[0]
                    page_id_clean = page_id.replace("-", "")
                    if __uuid_pattern.match(page_id_clean):
                        __system_affected_ids.add(page_id.lower())
                        print(f"[telemetry] tracked: {page_id.lower()}", file=__sys_sys.stderr)
                except (IndexError, AttributeError):
                    pass
        return __original_request(self, method, url, *args, **kwargs)

    __telemetry_request.__telemetry_patched__ = True
    __telemetry_request.__telemetry_original__ = __original_request
    __sys_requests.Session.request = __telemetry_request
""")

def _get_telemetry_header_with_map(resource_map: dict) -> str:
    import json as __sys_json
    resource_map_json = __sys_json.dumps(resource_map or {})
    return _TELEMETRY_HEADER + f"""
# --- resource map ---
RESOURCE_MAP = {resource_map_json}
"""


def _telemetry_footer(output_path: str) -> str:
    """Generate the footer code that dumps collected IDs to a JSON file."""
    return textwrap.dedent(f"""\

        # --- telemetry footer ---
        try:
            __sys_os.makedirs(__sys_os.path.dirname("{output_path}") or ".", exist_ok=True)
            with open("{output_path}", "w") as __f:
                __sys_json.dump(list(__system_affected_ids), __f)
        except Exception:
            pass
    """)


def wrap_code_with_telemetry(code: str, *, local: bool = False, resource_map: dict = None) -> str:
    """Wrap LLM-generated code with the telemetry prepend/append sandwich.

    Args:
        code: The raw Python code string produced by codegen.
        local: If True, use a workspace-relative output path instead of /tmp.
        resource_map: Mapping of page titles to IDs to inject into the execution scope.

    Returns:
        The instrumented code string ready for execution.
    """
    header = _get_telemetry_header_with_map(resource_map)
    output_path = LOCAL_AFFECTED_IDS_PATH if local else AFFECTED_IDS_PATH
    return f"{header}\n{code}\n{_telemetry_footer(output_path)}"
