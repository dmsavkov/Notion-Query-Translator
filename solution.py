import os
import dotenv
dotenv.load_dotenv()
import requests
from datetime import datetime


def get_project_id(project_name: str, notion_token: str, projects_database_id: str) -> str:
    """Retrieves the ID of a project from the Projects database.

    Args:
        project_name: The name of the project to search for.
        notion_token: The Notion API token.
        projects_database_id: The ID of the Projects database.

    Returns:
        The ID of the project if found, otherwise None.
    """
    url = f"https://api.notion.com/v1/databases/{projects_database_id}/query"
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    data = {
        "filter": {
            "property": "Name",
            "title": {
                "equals": project_name
            }
        }
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    results = response.json().get("results", [])
    if results:
        return results[0].get("id")
    else:
        print(f"Warning: Project '{project_name}' not found in Projects database.")
        return None


def add_task(title: str, date: str, importance: int, projects: list[str], notion_token: str = os.getenv('NOTION_TOKEN')) -> dict:
    """Adds a new task to the Tasks database.

    Args:
        title: The title of the task.
        date: The date of the task (YYYY-MM-DD).
        importance: The importance of the task (1-4).
        projects: A list of project names to associate with the task.
        notion_token: The Notion API token.

    Returns:
        The JSON response from the API containing the newly created task's information.

    Raises:
        requests.exceptions.HTTPError: If the API request fails.
    """
    tasks_database_id = os.getenv('NOTION_TASKS_DATABASE_ID')
    projects_database_id = os.getenv('NOTION_PROJECTS_DATABASE_ID')

    if not tasks_database_id:
        raise ValueError("NOTION_TASKS_DATABASE_ID not set in environment variables.")
    if not projects_database_id:
        raise ValueError("NOTION_PROJECTS_DATABASE_ID not set in environment variables.")

    project_ids = []
    for project_name in projects:
        project_id = get_project_id(project_name, notion_token, projects_database_id)
        if project_id:
            project_ids.append({"id": project_id})

    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    data = {
        "parent": {"database_id": tasks_database_id},
        "properties": {
            "Name": {"title": [{"text": {"content": title}} ]},
            "Importance": {"select": {"name": f"Importance {importance}"}},
            "Project": project_ids
        }
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status() Good good

    return response.json()


if __name__ == '__main__':
    # Example usage (replace with your actual values)
    try:
        task_info = add_task(
            title="Add_task_test",
            date="2024-01-01",
            importance=4,
            projects=["Agents & LLMs Specialization"]
        )
        print(task_info)
    except Exception as e:
        print(f"Error adding task: {e}")