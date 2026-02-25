import os
import pytest
from unittest.mock import patch, MagicMock
from solution import add_task

@pytest.fixture
def mock_requests_post():
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "id": "test-page-id"
        }
        yield mock_post

@pytest.fixture
def mock_os_environ():
    with patch.dict(os.environ, {
        'NOTION_TOKEN': 'test-token',
        'NOTION_TASKS_DATABASE_ID': 'test-database-id'
    }):
        yield

def test_add_task_happy_path(mock_requests_post, mock_os_environ):
    mock_requests_post.return_value.status_code = 200
    mock_requests_post.return_value.json.return_value = {
        "id": "test-page-id"
    }
    
    page_id = add_task("Test Task", "test-token", "test-database-id")
    
    assert page_id == "test-page-id"
    mock_requests_post.assert_called_once_with(
        'https://api.notion.com/v1/databases/test-database-id/pages',
        headers={'Authorization': 'Bearer test-token', 'Content-Type': 'application/json'},
        json={"Name": {"title": [{"text": {"content": "Test Task"}}]}, 
              "Importance": {"number": 4}, 
              "Due Date": {"date": {"start": "2026-02-22"}}}
    )

def test_add_task_api_error(mock_requests_post, mock_os_environ):
    mock_requests_post.return_value.status_code = 400
    mock_requests_post.return_value.json.return_value = {"error": "Test Error"}

    with pytest.raises(Exception) as excinfo:
        add_task("Test Task", "test-token", "test-database-id")

    assert "Test Error" in str(excinfo.value)
    mock_requests_post.assert_called_once_with(
        'https://api.notion.com/v1/databases/test-database-id/pages',
        headers={'Authorization': 'Bearer test-token', 'Content-Type': 'application/json'},
        json={"Name": {"title": [{"text": {"content": "Test Task"}}]}, 
              "Importance": {"number": 4}, 
              "Due Date": {"date": {"start": "2026-02-22"}}}
    )

def test_add_task_missing_fields(mock_requests_post, mock_os_environ):
    with pytest.raises(TypeError):
        add_task(None, "test-token", "test-database-id")

    with pytest.raises(TypeError):
        add_task("Test Task", None, "test-database-id")

    with pytest.raises(TypeError):
        add_task("Test Task", "test-token", None)