import pytest
from src.all_functionality import async_chat_wrapper

@pytest.mark.llm
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_chat_wrapper_real_gemma4():
    """
    Test real LLM call to gemma4 to ensure it returns valid JSON.
    This will be recorded by pytest-recording (vcrpy).
    """
    messages = [
        {"role": "user", "content": 'Return a JSON object with a "test" key and "passed" value.'}
    ]
    
    # Using gemma4 as requested
    result = await async_chat_wrapper(
        messages=messages,
        model_size="gemma4",
        json_output=True,
        max_tokens=100
    )
    
    assert isinstance(result, dict)
    assert result.get("test") in ["passed", "completed"]
