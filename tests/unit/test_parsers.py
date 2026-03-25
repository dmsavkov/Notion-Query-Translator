import pytest
import json
from src.all_functionality import extract_json_from_response, parse_statements_response

@pytest.mark.unit
def test_extract_json_from_response_success():
    # Strict JSON
    assert extract_json_from_response('{"key": "value"}') == {"key": "value"}
    
    # Markdown wrapped
    assert extract_json_from_response('```json\n{"key": "value"}\n```') == {"key": "value"}
    assert extract_json_from_response('```\n{"key": "value"}\n```') == {"key": "value"}
    
    # Conversational wrapper
    text = "Here is the response: \n { \"status\": \"success\", \"data\": [1, 2, 3] } \n Hope this helps!"
    assert extract_json_from_response(text) == {"status": "success", "data": [1, 2, 3]}

@pytest.mark.unit
def test_extract_json_from_response_failure():
    with pytest.raises(ValueError, match="Failed to extract JSON"):
        extract_json_from_response("This is just text with no json objects.")
    
    with pytest.raises(ValueError, match="Response content is None"):
        extract_json_from_response(None)

@pytest.mark.unit
def test_parse_statements_response():
    # Valid list input
    input_list = [{"statement": "s1", "status": "present"}, {"statement": "s2", "status": "wrong"}]
    assert parse_statements_response(input_list) == input_list
    
    # Raw string input with repair
    raw_str = '[{"statement": "s1", "status": "present"}, {"statement": "s2", "status": "wrong"}]'
    assert parse_statements_response(raw_str) == input_list
    
    # Filtering malformed objects
    mixed_input = [
        {"statement": "valid", "status": "present"},
        {"no_statement": "invalid"},
        {"statement": "no_status"}
    ]
    assert parse_statements_response(mixed_input) == [{"statement": "valid", "status": "present"}]
    
    # Empty/None input
    assert parse_statements_response(None) == []
    assert parse_statements_response("") == []
    assert parse_statements_response("Not a JSON array") == []
