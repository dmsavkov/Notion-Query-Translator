import os
import requests
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Inherit existing credentials before overriding with test instances
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

def get_database_schema(db_id: str) -> dict:
    """Fetch database schema and return property names and types"""
    url = f"https://api.notion.com/v1/databases/{db_id}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()
    
    properties = {}
    for prop_name, prop_config in data.get("properties", {}).items():
        properties[prop_name] = prop_config.get("type", "unknown")
    
    return properties

def delete_database(db_id: str) -> bool:
    """Delete a database"""
    url = f"https://api.notion.com/v1/databases/{db_id}"
    payload = {"archived": True}
    response = requests.patch(url, json=payload, headers=HEADERS)
    
    if response.status_code == 200:
        print(f"✓ Archived database: {db_id}")
        return True
    else:
        print(f"❌ Failed to archive database: {response.text}")
        return False

def add_property_to_database(db_id: str, property_name: str, property_config: dict) -> bool:
    """Add a property to an existing database"""
    url = f"https://api.notion.com/v1/databases/{db_id}"
    payload = {
        "properties": {
            property_name: property_config
        }
    }
    
    response = requests.patch(url, json=payload, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"❌ Failed to add property '{property_name}' to database")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    print(f"✓ Added property '{property_name}' to database")
    return True

def ensure_tasks_db_schema(db_id: str, projects_db_id: str):
    """Ensure tasks database has all required properties"""
    schema = get_database_schema(db_id)
    
    required_properties = {
        "Status": {
            "status": {}  # Don't specify options in PATCH, Notion will use defaults
        }
    }
    
    for prop_name, prop_config in required_properties.items():
        if prop_name not in schema:
            print(f"  Adding missing property: {prop_name}")
            add_property_to_database(db_id, prop_name, prop_config)

def recreate_inbox_page() -> str:
    """Archives existing Inbox pages to flush data, then creates a fresh node."""
    search_url = "https://api.notion.com/v1/search"
    search_payload = {
        "query": "Sandbox Inbox",
        "filter": {"value": "page", "property": "object"}
    }
    
    # Archive existing instances
    search_resp = requests.post(search_url, json=search_payload, headers=HEADERS)
    for page in search_resp.json().get("results", []):
        if not page.get("archived"):
            requests.patch(f"https://api.notion.com/v1/pages/{page['id']}", json={"archived": True}, headers=HEADERS)
    
    # Instantiate fresh Inbox
    create_url = "https://api.notion.com/v1/pages"
    create_payload = {
        "parent": {"type": "page_id", "page_id": PARENT_PAGE_ID},
        "properties": {
            "title": {"title": [{"text": {"content": "Sandbox Inbox"}}]}
        }
    }
    response = requests.post(create_url, json=create_payload, headers=HEADERS)
    response.raise_for_status()
    return response.json()["id"]

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
            "Due Date": {"date": {}},
            "Last Reviewed": {"date": {}}, 
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
        print(f"\n❌ ERROR creating tasks database")
        print(f"Status Code: {response.status_code}")
        print(f"Response body: {response.text}\n")
    else:
        # Check what properties were actually created
        created_db = response.json()
        created_properties = {k: v.get("type") for k, v in created_db.get("properties", {}).items()}
        print(f"\n✓ Database created with properties: {list(created_properties.keys())}")
        if "Status" not in created_properties:
            print(f"⚠️  WARNING: Status property NOT present in created database!")
            print(f"   This may indicate Notion rejected the property definition")
    
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
        print(f"\n❌ ERROR creating page")
        print(f"Status Code: {response.status_code}")
        print(f"Response body: {response.text}")
        print(f"Payload sent:\n{json.dumps(payload, indent=2)}\n")
    
    response.raise_for_status()
    return response.json()["id"]

def export_test_environment(db_ids: dict, page_ids: dict):
    """Outputs standard API credentials and dynamic testing IDs to ../.env"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    
    env_content = f"""# Notion API Configuration
NOTION_TOKEN={NOTION_TOKEN}

# Notion Database IDs
NOTION_PROJECTS_DATABASE_ID={db_ids['projects']}
NOTION_PROJECTS_DATA_SOURCE_ID={db_ids['projects']}
NOTION_TASKS_DATABASE_ID={db_ids['tasks']}
NOTION_TASKS_DATA_SOURCE_ID={db_ids['tasks']}
NOTION_INBOX_PAGE_ID={page_ids['inbox']}
ID_PROJECT_PAGE_ID={page_ids['id_project']}
ID_UPDATE_PAGE_ID={page_ids['id_update']}

# Google API Configuration
GOOGLE_API_KEY={os.environ.get("GOOGLE_API_KEY", "your_google_api_key_here")}

# LangSmith Configuration
LANGSMITH_TRACING={os.environ.get("LANGSMITH_TRACING", "true")}
LANGSMITH_ENDPOINT={os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")}
LANGSMITH_API_KEY={os.environ.get("LANGSMITH_API_KEY", "your_langsmith_api_key_here")}
LANGSMITH_PROJECT={os.environ.get("LANGSMITH_PROJECT", "your_project_name_here")}
"""
    with open(env_path, "w") as f:
        f.write(env_content)
    print(f"Test Environment deterministically generated at: {env_path}")

def provision_infrastructure():
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

    # Inspect available properties in tasks database
    print("\n📋 Inspecting Tasks Database Schema...")
    tasks_schema = get_database_schema(tasks_db_id)
    print(f"Available properties: {list(tasks_schema.keys())}")
    for prop_name, prop_type in sorted(tasks_schema.items()):
        print(f"  - {prop_name}: {prop_type}")

    print("\nFlushing state: Archiving existing Sandbox Inbox & emptying Databases...")
    inbox_page_id = recreate_inbox_page()
    flush_database(projects_db_id)
    flush_database(tasks_db_id)

    print("Seeding baseline determinism...")
    
    # 1. Target ID Project
    id_project_page_id = create_page(projects_db_id, {
        "Name": {"title": [{"text": {"content": "ID PROJECT PAGE"}}]}
    })

    # 2. Target ID Update Task
    id_update_page_id = create_page(tasks_db_id, {
        "Name": {"title": [{"text": {"content": "ID UPDATE PAGE"}}]},
        "Project": {"relation": [{"id": id_project_page_id}]}
    })

    # 3. Supplemental Seed: AI Research (For dynamic relation evaluation)
    ai_research_id = create_page(projects_db_id, {
        "Name": {"title": [{"text": {"content": "AI Research"}}]}
    })

    # 4. Supplemental Seed: Due in 3 days (Date Math Target)
    target_date = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    create_page(tasks_db_id, {
        "Name": {"title": [{"text": {"content": "Imminent Validation"}}]},
        "Due Date": {"date": {"start": target_date}},
        "Intensity": {"select": {"name": "3"}},
        "Project": {"relation": [{"id": ai_research_id}]}
    })

    print("Executing final state export...")
    export_test_environment(
        db_ids={"projects": projects_db_id, "tasks": tasks_db_id},
        page_ids={
            "inbox": inbox_page_id,
            "id_project": id_project_page_id,
            "id_update": id_update_page_id
        }
    )
    print("Provisioning Complete. Operational readiness achieved.")

if __name__ == "__main__":
    provision_infrastructure()