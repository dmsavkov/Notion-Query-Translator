import os
from unittest.mock import patch, MagicMock
import requests
import pytest

from solution import add_task

@pytest.fixture
def env_vars():
    with patch.dict(os.environ, {
        'NOTION_TOKEN': 'test-token',
        'NOTION_TASKS_DATABASE_ID': '0c0c2dea-6c50-4abb-b720-52f00d875899',
        'NOTION_PROJECTS_DATABASE_ID': 'cf09f790-71bb-4aac-8b51-5287d68ccca8',
    }):
        yield


def test_add_task_happy_path():
    mock_requests_post = MagicMock(return_value=requests.Response(status_code=200, json={'id': 'test-task-id'}))
    with patch('requests.post', mock_requests_post):
        result = add_task(
            title="Add_task_test",
            date="2024-01-01",
            importance=4,
            projects=["Agents & LLMs Specialization"],
            notion_token="test-token"
        )
        assert result == {'id': 'test-task-id'}
        mock_requests_post.assert_called_once_with(
            "https://api.notion.com/v1/pages",
            headers={
                "Authorization": "Bearer test-token",
                "Content-Type": "application/json",
                "Notion-Version": "2025-09-03"
            },
            json={
                "parent": {"database_id": "0c0c2dea-6c50-4abb-b720-52f00d875899"},
                "properties": {
                    "Name": {"title": [{"text": {"content": "Add_task_test"}} ]},
                    "Importance": {"select": {"name": "Importance 4"}},
                    "Project": [
                        {"id": "project-id-for-agents-llms"}
                    ]
                }
            }
        )


def test_add_task_api_error():
    mock_requests_post = MagicMock(return_value=requests.Response(status_code=400))
    with patch('requests.post', mock_requests_post):
        with pytest.raises(requests.exceptions.HTTPError) as excinfo:
            add_task(
                title="Add_task_test",
                date="2024-01-01",
                importance=4,
                projects=["Agents & LLMs Specialization"],
                notion_token="test-token"
            )
        assert excinfo.value.response.status_code == 400


