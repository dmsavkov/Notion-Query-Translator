from src.models.prompts import build_generate_code_prompt


def test_generate_code_prompt_omits_sandbox_page_ids_by_default() -> None:
    prompt = build_generate_code_prompt(
        general_info="<user_request>Test</user_request>",
        test_code="No tests",
    )

    assert "NOTION_INBOX_PAGE_ID" in prompt
    assert "NOTION_ID_PROJECT_PAGE_ID" not in prompt
    assert "NOTION_ID_UPDATE_PAGE_ID" not in prompt


def test_generate_code_prompt_can_include_sandbox_page_ids_when_enabled() -> None:
    prompt = build_generate_code_prompt(
        general_info="<user_request>Test</user_request>",
        test_code="No tests",
        prompt_pass_sandbox_id_notion_pages=True,
    )

    assert "NOTION_INBOX_PAGE_ID" in prompt
    assert "NOTION_ID_PROJECT_PAGE_ID" in prompt
    assert "NOTION_ID_UPDATE_PAGE_ID" in prompt
