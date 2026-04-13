"""Tests for the presentation layer: sanitization, viewer, telemetry, and sandbox integration.

Uses static JSON/markdown fixtures from tests/fixtures/ so no network
or sandbox infrastructure is required for unit tests.
Integration tests (marked with ``@pytest.mark.integration``) exercise
real E2B sandbox telemetry extraction.
"""

import json
import os
import re
from io import StringIO
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.presentation.sanitization import (
    IGNORE_PROPS,
    flatten_notion_properties,
    sanitize_notion_markdown,
)
from src.presentation.viewer import print_completed_state
from src.utils.page_cache import CachedPage
from src.utils.telemetry import (
    AFFECTED_IDS_PATH,
    LOCAL_AFFECTED_IDS_PATH,
    wrap_code_with_telemetry,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_page_properties() -> Dict[str, Any]:
    return json.loads((FIXTURES / "page_properties.json").read_text(encoding="utf-8"))


def _load_page_markdown() -> str:
    return (FIXTURES / "page_markdown.md").read_text(encoding="utf-8")


# ===========================================================================
# Sanitization — property flattening
# ===========================================================================

class TestFlattenProperties:
    """Verify property flattening against the fixture page."""

    @pytest.fixture()
    def flat(self) -> Dict[str, Any]:
        return flatten_notion_properties(_load_page_properties())

    def test_title_extracted(self, flat: Dict[str, Any]) -> None:
        assert flat.get("Name") == "Architecture Review"

    def test_status_extracted(self, flat: Dict[str, Any]) -> None:
        assert flat.get("Status") == "In Progress"

    def test_select_extracted(self, flat: Dict[str, Any]) -> None:
        assert flat.get("Priority") == "High"

    def test_multi_select_extracted(self, flat: Dict[str, Any]) -> None:
        tags = flat.get("Tags")
        assert isinstance(tags, list)
        assert "Backend" in tags
        assert "Architecture" in tags

    def test_date_extracted(self, flat: Dict[str, Any]) -> None:
        date_val = flat.get("Due Date")
        assert isinstance(date_val, dict)
        assert date_val.get("start") == "2026-04-15"

    def test_people_extracted(self, flat: Dict[str, Any]) -> None:
        people = flat.get("Assigned To")
        assert isinstance(people, list)
        assert people[0]["name"] == "Dmitry"

    def test_checkbox_extracted(self, flat: Dict[str, Any]) -> None:
        raw = _load_page_properties()
        full_flat = flatten_notion_properties(raw)
        # False is a meaningful value and should NOT be stripped
        assert "Done" in full_flat or full_flat.get("Done") is False

    def test_number_extracted(self, flat: Dict[str, Any]) -> None:
        assert flat.get("Score") == 85

    def test_relation_extracted(self, flat: Dict[str, Any]) -> None:
        rels = flat.get("Related Projects")
        assert isinstance(rels, list)
        assert len(rels) == 2
        assert "rel-page-id-001" in rels

    def test_rich_text_extracted(self, flat: Dict[str, Any]) -> None:
        desc = flat.get("Description")
        assert isinstance(desc, str)
        assert "Review the full system architecture" in desc

    def test_unique_id_extracted(self, flat: Dict[str, Any]) -> None:
        assert flat.get("Task ID") == "TASK-42"

    def test_url_extracted(self, flat: Dict[str, Any]) -> None:
        assert flat.get("URL") == "https://github.com/example/repo"

    def test_page_id_present(self, flat: Dict[str, Any]) -> None:
        assert "page_id" in flat

    def test_ignored_props_set_exists(self) -> None:
        assert "page_object" in IGNORE_PROPS
        assert "page_archived" in IGNORE_PROPS

    def test_empty_input_returns_empty(self) -> None:
        assert flatten_notion_properties({}) == {}
        assert flatten_notion_properties(None) == {}  # type: ignore


# ===========================================================================
# Sanitization — markdown cleanup
# ===========================================================================

class TestSanitizeMarkdown:
    """Verify markdown sanitization transformations."""

    @pytest.fixture()
    def clean(self) -> str:
        return sanitize_notion_markdown(_load_page_markdown())

    def test_summary_tag_converted(self, clean: str) -> None:
        assert "**► Core Engine Design**" in clean

    def test_callout_tag_converted(self, clean: str) -> None:
        assert "> 💡 Always validate" in clean

    def test_page_link_converted(self, clean: str) -> None:
        assert "[Related Design Document](https://notion.so/some-page-id)" in clean

    def test_database_link_converted(self, clean: str) -> None:
        assert "[Task Tracker](https://notion.so/db-id)" in clean

    def test_mention_page_converted(self, clean: str) -> None:
        assert "[Referenced](https://notion.so/ref-page)" in clean

    def test_html_table_converted(self, clean: str) -> None:
        assert "| Step | Component | Output |" in clean
        assert "| 1 | Precheck | Safety verdict |" in clean

    def test_details_tag_stripped(self, clean: str) -> None:
        assert "<details" not in clean
        assert "</details>" not in clean

    def test_empty_block_stripped(self, clean: str) -> None:
        assert "<empty-block" not in clean

    def test_code_block_preserved(self, clean: str) -> None:
        assert "def example():" in clean

    def test_blockquote_preserved(self, clean: str) -> None:
        assert "> **Note:**" in clean

    def test_none_input(self) -> None:
        assert sanitize_notion_markdown(None) == ""

    def test_dict_input(self) -> None:
        result = sanitize_notion_markdown({"markdown": "# Hello"})
        assert "# Hello" in result


# ===========================================================================
# Telemetry — code wrapping
# ===========================================================================

class TestTelemetryWrapping:
    """Verify the telemetry sandwich produces correct, executable Python code."""

    def test_wrap_adds_header_and_footer(self) -> None:
        code = "print('hello')"
        wrapped = wrap_code_with_telemetry(code)
        assert "__system_read_ids" in wrapped
        assert "__system_mutated_ids" in wrapped
        assert "__telemetry_request" in wrapped
        assert "/tmp/affected_ids.json" in wrapped
        assert code in wrapped

    def test_wrap_local_uses_local_path(self) -> None:
        code = "print('hello')"
        wrapped = wrap_code_with_telemetry(code, local=True)
        assert "data/tmp_affected_ids.json" in wrapped
        assert "/tmp/affected_ids.json" not in wrapped

    def test_regex_quantifiers_are_valid(self) -> None:
        """The UUID regex must use single-brace quantifiers {8}, not {{8}}."""
        wrapped = wrap_code_with_telemetry("pass")
        # The regex pattern in the output code must have {8}, {4}, {12}
        assert "{8}" in wrapped, "Regex quantifier {8} missing — likely double-brace bug"
        assert "{4}" in wrapped, "Regex quantifier {4} missing"
        assert "{12}" in wrapped, "Regex quantifier {12} missing"
        # Must NOT have the double-brace variant
        assert "{{8}}" not in wrapped, "Double-brace {{8}} found — regex will fail"

    def test_wrapped_code_is_syntactically_valid(self) -> None:
        """The wrapped code must compile without SyntaxError."""
        wrapped = wrap_code_with_telemetry("x = 1 + 1")
        compile(wrapped, "<telemetry_test>", "exec")  # raises SyntaxError on failure

    def test_wrap_is_reentrant_across_execs(self) -> None:
        """A reused interpreter must still capture telemetry on later executions."""
        page_id = "ccbcb17d-cc44-829a-b707-019899e91df7"
        code = f"""
import requests
requests.get("https://api.notion.com/v1/pages/{page_id}")
"""
        wrapped = wrap_code_with_telemetry(code)
        wrapped = wrapped.replace(
            "return __original_request(self, method, url, *args, **kwargs)",
            "return type('R', (), {'status_code': 200, 'json': lambda self: {}, 'text': ''})()",
        )

        namespace: Dict[str, Any] = {}
        exec(wrapped, namespace)
        first_ids = list(namespace.get("__system_read_ids", set()))
        exec(wrapped, namespace)
        second_ids = list(namespace.get("__system_read_ids", set()))

        assert page_id in first_ids
        assert page_id in second_ids

    def test_uuid_regex_matches_notion_ids(self) -> None:
        """Verify the compiled regex actually matches Notion-style UUIDs."""
        # Extract the regex pattern from the wrapped code
        wrapped = wrap_code_with_telemetry("pass")
        # Find the regex string
        pattern_match = re.search(r"r'([^']+)'", wrapped)
        assert pattern_match, "Could not find regex pattern in wrapped code"
        pattern = re.compile(pattern_match.group(1))

        # Test with hyphenated and non-hyphenated UUIDs
        assert pattern.match("ccbcb17d-cc44-829a-b707-019899e91df7")
        assert pattern.match("ccbcb17dcc44829ab707019899e91df7")
        assert not pattern.match("not-a-uuid-at-all")
        assert not pattern.match("zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz")


# ===========================================================================
# Telemetry — ID extraction from mock payloads
# ===========================================================================

class TestTelemetryExtraction:
    """Verify that telemetry correctly extracts IDs from various URL patterns."""

    def _run_telemetry_code(self, urls_and_methods: list[tuple[str, str]]) -> dict[str, list[str]]:
        """Execute the telemetry header + simulated requests and return captured IDs."""
        # Build test code that simulates requests to the given URLs
        lines = []
        for method, url in urls_and_methods:
            lines.append(f"__sys_requests.{method.lower()}('{url}')")

        test_code = "\n".join(lines)
        full_code = wrap_code_with_telemetry(test_code)

        # Replace the actual HTTP call with a no-op to avoid network access
        full_code = full_code.replace(
            "return __original_request(self, method, url, *args, **kwargs)",
            "return type('R', (), {'status_code': 200, 'json': lambda self: {}, 'text': ''})()"
        )

        # Execute in isolated namespace
        namespace: Dict[str, Any] = {}
        exec(full_code, namespace)
        read_ids = list(namespace.get("__system_read_ids", set()))
        mutated_ids = list(namespace.get("__system_mutated_ids", set()))
        return {"read": read_ids, "mutated": mutated_ids}

    def test_extracts_page_id_from_patch(self) -> None:
        ids = self._run_telemetry_code([
            ("PATCH", "https://api.notion.com/v1/pages/ccbcb17d-cc44-829a-b707-019899e91df7"),
        ])
        assert "ccbcb17d-cc44-829a-b707-019899e91df7" in ids["mutated"]

    def test_extracts_page_id_from_get(self) -> None:
        ids = self._run_telemetry_code([
            ("GET", "https://api.notion.com/v1/pages/aabbccdd-1122-3344-5566-778899aabbcc"),
        ])
        assert "aabbccdd-1122-3344-5566-778899aabbcc" in ids["read"]

    def test_extracts_from_blocks_endpoint(self) -> None:
        ids = self._run_telemetry_code([
            ("PATCH", "https://api.notion.com/v1/blocks/aabbccdd-1122-3344-5566-778899aabbcc/children"),
        ])
        assert "aabbccdd-1122-3344-5566-778899aabbcc" in ids["mutated"]

    def test_deduplicates_multiple_calls(self) -> None:
        page_id = "ccbcb17d-cc44-829a-b707-019899e91df7"
        ids = self._run_telemetry_code([
            ("GET", f"https://api.notion.com/v1/pages/{page_id}"),
            ("PATCH", f"https://api.notion.com/v1/pages/{page_id}"),
        ])
        assert ids["read"].count(page_id) == 1
        assert ids["mutated"].count(page_id) == 1

    def test_ignores_non_notion_urls(self) -> None:
        ids = self._run_telemetry_code([
            ("GET", "https://example.com/v1/pages/ccbcb17d-cc44-829a-b707-019899e91df7"),
        ])
        assert len(ids["read"]) == 0
        assert len(ids["mutated"]) == 0

    def test_extracts_unhyphenated_id(self) -> None:
        ids = self._run_telemetry_code([
            ("GET", "https://api.notion.com/v1/pages/ccbcb17dcc44829ab707019899e91df7"),
        ])
        assert "ccbcb17dcc44829ab707019899e91df7" in ids["read"]


# ===========================================================================
# Telemetry — file output
# ===========================================================================

class TestTelemetryFileOutput:
    """Verify that the footer writes the JSON file correctly."""

    def test_footer_writes_json_file(self, tmp_path: Path) -> None:
        output_file = tmp_path / "affected_ids.json"
        code = f"""
import requests
requests.get("https://api.notion.com/v1/pages/ccbcb17d-cc44-829a-b707-019899e91df7")
"""
        wrapped = wrap_code_with_telemetry(code)
        # Redirect the output to our temp path (use forward slashes for cross-platform exec)
        wrapped = wrapped.replace(AFFECTED_IDS_PATH, str(output_file).replace("\\", "/"))
        # Mock the actual HTTP call
        wrapped = wrapped.replace(
            "return __original_request(self, method, url, *args, **kwargs)",
            "return type('R', (), {'status_code': 200, 'json': lambda self: {}, 'text': ''})()"
        )

        exec(wrapped, {})

        assert output_file.exists(), "Telemetry footer did not write the JSON file"
        payload = json.loads(output_file.read_text())
        assert isinstance(payload, dict)
        assert "ccbcb17d-cc44-829a-b707-019899e91df7" in payload.get("read", [])


# ===========================================================================
# Viewer — state-based rendering
# ===========================================================================

class TestViewer:
    """Verify viewer output for various terminal states."""

    def test_security_blocked_renders_panel(self, capsys: pytest.CaptureFixture) -> None:
        """Security-blocked state must show a clear rejection message."""
        state = {
            "terminal_status": "security_blocked",
            "execution_output": "Blocked by LlamaGuard.",
            "affected_notion_ids": [],
        }
        print_completed_state(state)
        captured = capsys.readouterr().out
        assert "Security Block" in captured or "security" in captured.lower()
        assert "blocked" in captured.lower()

    def test_execution_failed_renders_panel(self, capsys: pytest.CaptureFixture) -> None:
        """Execution failure must show the error and feedback."""
        state = {
            "terminal_status": "execution_failed",
            "execution_output": "NameError: undefined var",
            "feedback": "Variable was not defined.",
            "affected_notion_ids": [],
        }
        print_completed_state(state)
        captured = capsys.readouterr().out
        assert "Execution Failed" in captured or "failed" in captured.lower()
        assert "NameError" in captured

    def test_max_retries_exceeded_renders_panel(self, capsys: pytest.CaptureFixture) -> None:
        """Max retries exceeded must show the failure state distinctly."""
        state = {
            "terminal_status": "max_retries_exceeded",
            "execution_output": "Timeout after 3 attempts",
            "feedback": "Code kept timing out.",
            "affected_notion_ids": [],
        }
        print_completed_state(state)
        captured = capsys.readouterr().out
        assert "max_retries_exceeded" in captured or "Failed" in captured

    def test_success_no_ids_shows_output(self, capsys: pytest.CaptureFixture) -> None:
        """Success without affected IDs should show raw execution output."""
        state = {
            "terminal_status": "success",
            "execution_output": "Done successfully.",
            "affected_notion_ids": [],
        }
        print_completed_state(state)
        captured = capsys.readouterr().out
        assert "Done successfully" in captured

    @patch("src.presentation.viewer.fetch_page_properties")
    @patch("src.presentation.viewer.fetch_page_markdown")
    def test_success_with_ids_renders_pages(
        self,
        mock_md: Any,
        mock_props: Any,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Success with affected IDs should fetch and render each page."""
        mock_props.return_value = _load_page_properties()
        mock_md.return_value = _load_page_markdown()

        state = {
            "terminal_status": "success",
            "execution_output": "",
            "affected_notion_ids": ["ccbcb17d-cc44-829a-b707-019899e91df7"],
        }
        print_completed_state(state)
        captured = capsys.readouterr().out
        assert "Architecture Review" in captured
        assert "Page Properties" in captured

    @patch("src.presentation.viewer.fetch_page_properties")
    @patch("src.presentation.viewer.fetch_page_markdown")
    def test_prefetched_pages_skip_fetch(
        self,
        mock_md: Any,
        mock_props: Any,
        capsys: pytest.CaptureFixture,
    ) -> None:
        mock_props.return_value = _load_page_properties()
        mock_md.return_value = _load_page_markdown()

        state = {
            "terminal_status": "success",
            "execution_output": "",
            "affected_notion_ids": ["ccbcb17d-cc44-829a-b707-019899e91df7"],
        }

        prefetched = {
            "any": CachedPage(
                page_id="ccbcb17d-cc44-829a-b707-019899e91df7",
                properties=_load_page_properties(),
                markdown=_load_page_markdown(),
            )
        }
        print_completed_state(state, prefetched=prefetched)
        captured = capsys.readouterr().out
        assert "Architecture Review" in captured
        mock_props.assert_not_called()
        mock_md.assert_not_called()

    @patch("src.presentation.viewer.fetch_page_properties")
    @patch("src.presentation.viewer.fetch_page_markdown")
    def test_fetch_failure_shows_error_panel(
        self,
        mock_md: Any,
        mock_props: Any,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """If fetching a page fails, show an error panel instead of crashing."""
        mock_props.side_effect = Exception("HTTP 404: Page not found")

        state = {
            "terminal_status": "success",
            "execution_output": "",
            "affected_notion_ids": ["nonexistent-page-id"],
        }
        print_completed_state(state)
        captured = capsys.readouterr().out
        assert "Failed to fetch" in captured or "404" in captured


# ===========================================================================
# Sandbox integration (requires E2B_API_KEY)
# ===========================================================================

@pytest.mark.integration
class TestSandboxTelemetryIntegration:
    """Real E2B sandbox tests for telemetry extraction.

    These tests require E2B_API_KEY and NOTION_TOKEN environment variables.
    Run with: pytest -m integration
    """

    @pytest.fixture(autouse=True)
    def _require_e2b_key(self) -> None:
        if not os.getenv("E2B_API_KEY"):
            pytest.skip("E2B_API_KEY not set")

    def test_sandbox_telemetry_captures_ids(self) -> None:
        """Full end-to-end: wrap code, run in sandbox, extract IDs from file."""
        from e2b_code_interpreter import Sandbox

        # Code that makes a real GET request to a known Notion page
        notion_token = os.environ.get("NOTION_TOKEN", "")
        if not notion_token:
            pytest.skip("NOTION_TOKEN not set")

        page_id = os.environ.get("NOTION_ID_PROJECT_PAGE_ID", "")
        if not page_id:
            pytest.skip("NOTION_ID_PROJECT_PAGE_ID not set")

        test_code = f"""
import requests
import os

headers = {{
    "Authorization": f"Bearer {{os.environ.get('NOTION_TOKEN', '')}}",
    "Notion-Version": "2022-06-28",
}}
resp = requests.get(
    "https://api.notion.com/v1/pages/{page_id}",
    headers=headers,
    timeout=15,
)
print(f"Status: {{resp.status_code}}")
"""
        instrumented = wrap_code_with_telemetry(test_code, local=False)

        sandbox = Sandbox.create(
            envs={k: v for k, v in os.environ.items() if k.startswith("NOTION_")},
            timeout=60,
        )
        try:
            execution = sandbox.run_code(instrumented, timeout=30)

            # Read back the telemetry file
            file_bytes = sandbox.files.read(AFFECTED_IDS_PATH)
            payload = json.loads(file_bytes)

            assert isinstance(payload, dict)
            read_ids = payload.get("read", [])
            assert isinstance(read_ids, list)
            assert page_id.lower().replace("-", "") in "".join(read_ids).replace("-", ""), (
                f"Expected {page_id} in captured IDs but got: {payload}"
            )
        finally:
            sandbox.kill()
