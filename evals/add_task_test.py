import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import the target function from the solution file
import sys
sys.path.append(r"D:\Coding\VSCFiles\IndependentProjects\AI\notion-query")
from solution import add_task

# --- FIXTURES ---

@pytest.fixture
def mock_dependencies():
    """
    Mocks requests.post to intercept the API call.
    """
    with patch("solution.requests.post") as mock_post:
        
        # Setup mock return values
        mock_post.return_value = MagicMock(json=lambda: {"object": "page", "id": "dummy_page_id"})
        
        yield mock_post


def soft_assert(condition: bool, message: str, failures: list):
    """Collect assertion failures without raising immediately."""
    if not condition:
        failures.append(message)
    return failures


# --- TEST CASES ---

def test_match_headers(mock_dependencies):
    """Verifies that the correct Notion API headers are provided."""
    failures = []
    
    # Provide necessary inputs
    add_task("dummy_database_id", "Add_task_test", 4, "31bcb17dcc4480dcb042f86e6a70bb70")
    
    # Extract the intercepted request parameters
    mock_post = mock_dependencies
    headers = mock_post.call_args.kwargs.get("headers", {})
    
    # Soft Assertions
    soft_assert("Authorization" in headers, "Missing Authorization header", failures)
    soft_assert(headers.get("Authorization", "").startswith("Bearer "), "Authorization must use Bearer token", failures)
    soft_assert(headers.get("Content-Type") == "application/json", "Content-Type must be application/json", failures)
    soft_assert(headers.get("Notion-Version") == "2022-06-28", "Missing or incorrect Notion-Version header", failures)
    
    # Report all failures at once
    assert not failures, "\n".join(failures)

def test_match_endpoint(mock_dependencies):
    """Verifies that the request hits the correct API endpoint."""
    failures = []
    
    add_task("dummy_database_id", "Add_task_test", 4, "31bcb17dcc4480dcb042f86e6a70bb70")
    
    mock_post = mock_dependencies
    
    # Check positional args (args[0]) or keyword args (kwargs['url'])
    args = mock_post.call_args.args
    url = args[0] if args else mock_post.call_args.kwargs.get("url")
    
    soft_assert(url == "https://api.notion.com/v1/pages", "Incorrect Notion endpoint for creating a page", failures)
    
    assert not failures, "\n".join(failures)

def test_match_api_method(mock_dependencies):
    """Verifies that the correct HTTP method (POST) is used."""
    failures = []
    
    add_task("dummy_database_id", "Add_task_test", 4, "31bcb17dcc4480dcb042f86e6a70bb70")
    
    mock_post = mock_dependencies
    
    # Soft Assertions
    soft_assert(mock_post.called, "The API method must be POST (requests.post)", failures)
    soft_assert(mock_post.call_count == 1, "Expected exactly one POST request", failures)
    
    assert not failures, "\n".join(failures)

def test_match_properties_schema(mock_dependencies):
    """Verifies that the JSON payload scrupulously matches the Notion API nested schema."""
    failures = []
    
    title_input = "Add_task_test"
    db_id_input = "dummy_database_id"
    
    add_task(db_id_input, title_input, 4, "31bcb17dcc4480dcb042f86e6a70bb70")
    
    mock_post = mock_dependencies
    payload = mock_post.call_args.kwargs.get("json", {})
    
    # 1. Check Parent Schema
    soft_assert("parent" in payload, "Payload missing 'parent' key", failures)
    soft_assert(payload.get("parent", {}).get("database_id") == db_id_input, "Incorrect or missing database_id in parent", failures)
    
    # 2. Check General Properties Schema
    soft_assert("properties" in payload, "Payload missing 'properties' key", failures)
    props = payload.get("properties", {})
    print(props)
    
    # 3. Check Title Schema (Rich Text Nesting)
    soft_assert("Name" in props, "Missing 'Name' property", failures)
    soft_assert("title" in props.get("Name", {}), "'Name' must contain 'title' key", failures)
    soft_assert(isinstance(props.get("Name", {}).get("title"), list), "'title' must be a list", failures)
    soft_assert(
        props.get("Name", {}).get("title", [{}])[0].get("text", {}).get("content") == title_input,
        "Title text content incorrectly mapped",
        failures
    )
    
    # 4. Check Date Schema
    soft_assert("Due Date" in props, "Missing 'Due Date' property", failures)
    soft_assert("date" in props.get("Due Date", {}), "'Due Date' must contain 'date' key", failures)
    soft_assert("start" in props.get("Due Date", {}).get("date", {}), "Date object missing 'start' key", failures)
    soft_assert(
        props.get("Due Date", {}).get("date", {}).get("start") == datetime.now().strftime("%Y-%m-%d"),
        "Date 'start' incorrectly formatted",
        failures
    )
    
    # 5. Check Select Schema
    soft_assert("Importance" in props, "Missing 'Importance' property", failures)
    soft_assert("select" in props.get("Importance", {}), "'Importance' must contain 'select' key", failures)
    soft_assert(
        props.get("Importance", {}).get("select", {}).get("name") == "4",
        "Select name must be a string matching the importance value",
        failures
    )
    
    # 6. Check Relation Schema
    soft_assert("Project" in props, "Missing 'Project' property", failures)
    soft_assert("relation" in props.get("Project", {}), "'Project' must contain 'relation' key", failures)
    soft_assert(isinstance(props.get("Project", {}).get("relation"), list), "Relation must be a list of objects", failures)
    soft_assert(
        "id" in props.get("Project", {}).get("relation", [{}])[0],
        "Relation items must contain an 'id' key",
        failures
    )
    
    assert not failures, "\n".join(failures)