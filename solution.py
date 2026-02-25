import os
import dotenv
dotenv.load_dotenv()
import requests


def add_task(task_name: str, notion_token: str, database_id: str) -> str:
    """Adds a task to the Tasks database in Notion.

    Args:
        task_name: The name of the task.
        notion_token: The Notion API token.
        database_id: The ID of the Tasks database.

    Returns:
        The ID of the newly created page.

    Raises:
        TypeError: If any of the input arguments are None.
        Exception: If the API request fails.
    """
    if task_name is None or notion_token is None or database_id is None:
        raise TypeError("All arguments must be provided.")

    headers = {
        "Notion-Version": "2025-09-03",
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json"
    }

    request_body = {
        "parent": {
            "database_id": database_id,
            "type": "database_id"
        },
        "properties": {
            "Name": {
                "title": [
                    {
                        "text": {
                            "content": task_name
                        }
                    }
                ]
            },
            "Importance": {
                "number": 4
            },
            "Due Date": {
                "date": {
                    "start": "2026-02-22"
                }
            }
        }
    }

    endpoint_url = f"https://api.notion.com/v1/data_sources"

    try:
        response = requests.post(endpoint_url, headers=headers, json=request_body)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        return data["id"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")
    except KeyError as e:
        raise Exception(f"Failed to parse API response: {e}")

if __name__ == "__main__":
    add_task("TESTING RAG", os.getenv("NOTION_TOKEN"), os.getenv("NOTION_TASKS_DATA_SOURCE_ID"))