from src.models.prompts import build_generate_code_prompt


def test_generate_code_prompt_omits_sandbox_page_ids_by_default() -> None:
    prompt = build_generate_code_prompt(
        general_info="<user_request>Test</user_request>",
        test_code="No tests",
    )

    assert "NOTION_INBOX_PAGE_ID" in prompt
    assert "NOTION_ID_PROJECT_PAGE_ID" not in prompt
    assert "NOTION_ID_UPDATE_PAGE_ID" not in prompt
    assert '"execution_status": "success|error"' in prompt
    assert '"message_to_user": "..."' in prompt
    assert '"relevant_page_ids": ["uuid-1", "uuid-2"]' in prompt
    assert "rendering is capped server-side" in prompt
    assert "The final stdout line must be the execution envelope JSON" in prompt


def test_generate_code_prompt_can_include_sandbox_page_ids_when_enabled() -> None:
    prompt = build_generate_code_prompt(
        general_info="<user_request>Test</user_request>",
        test_code="No tests",
        prompt_pass_sandbox_id_notion_pages=True,
    )

    assert "NOTION_INBOX_PAGE_ID" in prompt
    assert "NOTION_ID_PROJECT_PAGE_ID" in prompt
    assert "NOTION_ID_UPDATE_PAGE_ID" in prompt


def test_generate_code_prompt_accepts_custom_render_cap_text() -> None:
    prompt = build_generate_code_prompt(
        general_info="<user_request>Test</user_request>",
        test_code="No tests",
        max_rendered_relevant_page_ids=9,
    )

    assert "If more than 9 IDs are relevant" in prompt
