import os
import dotenv
import requests
from datetime import date

dotenv.load_dotenv()

NOTION_TOKEN = os.getenv('NOTION_TOKEN')
NOTION_TASKS_DATABASE_ID = os.getenv('NOTION_TASKS_DATABASE_ID')
NOTION_PROJECTS_DATABASE_ID = os.getenv('NOTION_PROJECTS_DATABASE_ID')

def add_task(title: str, due_date: date, importance: int, project_names: list[str]) -> str | None:
    """Adds a new task to the Notion Tasks database.

    Args:
        title: The title of the task.
        due_date: The due date of the task.
        importance: The importance of the task (1-4).
        project_names: A list of project names to associate with the task.

    Returns:
        The ID of the newly created task page, or None on failure.
    """
    if not 1 <= importance <= 4:
        raise ValueError("Importance must be between 1 and 4.")

    project_ids = []
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2025-09-03",
        "Content-Type": "application/json"
    }

    for project_name in project_names:
        query = {
            "filter": {
                "property": "Name",
                "title": {
                    "equals": project_name
                }
            }
        }
        url = f"https://api.notion.com/v1/databases/{NOTION_PROJECTS_DATABASE_ID}/query"
        response = requests.post(url, headers=headers, json=query)
        response.raise_for_status()
        data = response.json()

        if data["results"]:
            project_ids.append(data["results"][0]["id"])
        else:
            print(f"Project '{project_name}' not found.")
            return None

    url = f"https://api.notion.com/v1/pages"
    request_body = {
        "parent": {
            "database_id": NOTION_TASKS_DATABASE_ID
        },
        "properties": {
            "Name": {
                "title": [
                    {
                        "text": {
                            "content": title
                        }
                    }
                ]
            },
            "Due Date": {
                "date": {
                    "start": due_date.isoformat()
                }
            },
            "Importance": {
                "select": {
                    "name": str(importance)
                }
            },
            "Project": {
                "relation": [
                    {
                        "id": project_id
                    } for project_id in project_ids
                ]
            }
        }
    }

    try:
        response = requests.post(url, headers=headers, json=request_body)
        response.raise_for_status()
        data = response.json()
        return data["id"]
    except requests.exceptions.RequestException as e:
        print(f"Error creating task: {e}")
        return None


if __name__ == '__main__':
    today = date.today()
    task_id = add_task(
        title="Add_task_test",
        due_date=today,
        importance=4,
        project_names=["Agents & LLMs Specialization"]
    )
    if task_id:
        print(f"Task created with ID: {task_id}")
    else:
        print("Task creation failed.")