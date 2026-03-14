import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
PARENT_PAGE_ID = "322cb17dcc448041af19eefbeb7abd30"
HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

def search_database(title: str) -> str:
    url = "https://api.notion.com/v1/search"
    payload = {
        "query": title,
        "filter": {"value": "database", "property": "object"}
    }
    response = requests.post(url, json=payload, headers=HEADERS)
    response.raise_for_status()
    results = response.json().get("results", [])
    for db in results:
        if db["title"][0]["plain_text"] == title:
            return db["id"]
    return None

def create_projects_db() -> str:
    url = "https://api.notion.com/v1/databases"
    payload = {
        "parent": {"type": "page_id", "page_id": PARENT_PAGE_ID},
        "title": [{"type": "text", "text": {"content": "Sandbox Projects"}}],
        "properties": {
            "Name": {"title": {}}
        }
    }
    response = requests.post(url, json=payload, headers=HEADERS)
    response.raise_for_status()
    return response.json()["id"]

def create_tasks_db(projects_db_id: str) -> str:
    url = "https://api.notion.com/v1/databases"
    payload = {
        "parent": {"type": "page_id", "page_id": PARENT_PAGE_ID},
        "title": [{"type": "text", "text": {"content": "Sandbox Tasks"}}],
        "properties": {
            "Name": {"title": {}},
            "Status": {
                "status": {}
            },
            "Due Date": {"date": {}},
            "Due Date": {"date": {}}, 
            "Importance": {
                "select": {
                    "options": [{"name": "1"}, {"name": "2"}, {"name": "3"}, {"name": "4"}]
                }
            },
            "Urgency": {
                "select": {
                    "options": [{"name": "1"}, {"name": "2"}, {"name": "3"}, {"name": "4"}]
                }
            },
            "Intensity": {
                "select": {
                    "options": [{"name": "1"}, {"name": "2"}, {"name": "3"}, {"name": "5"}, {"name": "8"}]
                }
            },
            "Do Now": {"checkbox": {}},
            "Project": {
                "relation": {
                    "database_id": projects_db_id,
                    "type": "dual_property",
                    "dual_property": {}
                }
            }
        }
    }
    response = requests.post(url, json=payload, headers=HEADERS)
    if response.status_code != 200:
        print(f"Error Response: {response.status_code}")
        print(f"Response Body: {response.text}")
    response.raise_for_status()
    return response.json()["id"]

def flush_database(db_id: str):
    url = f"https://api.notion.com/v1/databases/{db_id}/query"
    response = requests.post(url, headers=HEADERS)
    response.raise_for_status()
    pages = response.json().get("results", [])
    
    for page in pages:
        patch_url = f"https://api.notion.com/v1/pages/{page['id']}"
        requests.patch(patch_url, json={"archived": True}, headers=HEADERS)

def create_page(db_id: str, properties: dict) -> str:
    url = "https://api.notion.com/v1/pages"
    payload = {
        "parent": {"database_id": db_id},
        "properties": properties
    }
    response = requests.post(url, json=payload, headers=HEADERS)
    if response.status_code != 200:
        print(f"Error Response: {response.status_code}")
        print(f"Response Body: {response.text}")
    response.raise_for_status()
    return response.json()["id"]

def provision_infrastructure():
    # 1. Verification & Instantiation Phase
    print("Validating Projects Database...")
    projects_db_id = search_database("Sandbox Projects")
    if not projects_db_id:
        projects_db_id = create_projects_db()
        print(f"Created Projects DB: {projects_db_id}")
    else:
        print(f"Located Projects DB: {projects_db_id}")

    print("Validating Tasks Database...")
    tasks_db_id = search_database("Sandbox Tasks")
    if not tasks_db_id:
        tasks_db_id = create_tasks_db(projects_db_id)
        print(f"Created Tasks DB: {tasks_db_id}")
    else:
        print(f"Located Tasks DB: {tasks_db_id}")

    # 2. Archival Phase (Flushing)
    print("Flushing existing data states...")
    flush_database(projects_db_id)
    flush_database(tasks_db_id)

    # 3. Deterministic Seeding Phase
    print("Seeding baseline determinism...")
    
    # Target Project
    ai_research_id = create_page(projects_db_id, {
        "Name": {"title": [{"text": {"content": "AI Research"}}]}
    })

    # Task: Due in 3 days (Date Math Target)
    target_date = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    create_page(tasks_db_id, {
        "Name": {"title": [{"text": {"content": "Imminent Validation"}}]},
        "Due Date": {"date": {"start": target_date}},
        "Intensity": {"select": {"name": "3"}},
        "Project": {"relation": [{"id": ai_research_id}]}
    })

    # Task: Stale state (Stale Task Target)
    stale_date = (datetime.now() - timedelta(days=31)).strftime("%Y-%m-%d")
    create_page(tasks_db_id, {
        "Name": {"title": [{"text": {"content": "Deprecated Node"}}]},
        "Due Date": {"date": {"start": stale_date}},
        "Status": {"status": {"name": "In progress"}}
    })

    # Task: Emergency (Vague Property Target)
    create_page(tasks_db_id, {
        "Name": {"title": [{"text": {"content": "Critical Overflow"}}]},
        "Status": {"status": {"name": "Not started"}},
        "Urgency": {"select": {"name": "4"}},
        "Do Now": {"checkbox": False}
    })

    print("Provisioning Complete. Operational readiness achieved.")

if __name__ == "__main__":
    provision_infrastructure()