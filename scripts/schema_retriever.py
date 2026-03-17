import os
import requests
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
PROJECTS_DB_ID = os.getenv("NOTION_PROJECTS_DATABASE_ID")
TASKS_DB_ID = os.getenv("NOTION_TASKS_DATABASE_ID")

HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}


def get_database_schema(db_id: str) -> Dict[str, Any]:
    """Fetch database schema"""
    url = f"https://api.notion.com/v1/databases/{db_id}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def get_python_type_for_property(prop_type: str, prop_config: Dict[str, Any]) -> str:
    """Generate Python type hint for a property"""
    if prop_type == "title" or prop_type == "text" or prop_type == "rich_text":
        return "str"
    elif prop_type == "number":
        return "float | int"
    elif prop_type == "select":
        options = prop_config.get("select", {}).get("options", [])
        option_names = [opt.get("name") for opt in options]
        return f"Literal[{', '.join(repr(opt) for opt in option_names)}]"
    elif prop_type == "multi_select":
        options = prop_config.get("multi_select", {}).get("options", [])
        option_names = [opt.get("name") for opt in options]
        return f"list[Literal[{', '.join(repr(opt) for opt in option_names)}]]"
    elif prop_type == "date":
        return "str | dict[str, str]"
    elif prop_type == "checkbox":
        return "bool"
    elif prop_type == "relation":
        return "list[str]"
    elif prop_type == "people":
        return "list[dict]"
    elif prop_type == "files":
        return "list[dict]"
    elif prop_type in ("url", "email", "phone_number"):
        return "str"
    else:
        return "Any"


def extract_property_info(prop_name: str, prop_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract detailed information about a property including its possible values"""
    prop_type = prop_config.get("type", "unknown")
    info = {
        "name": prop_name,
        "type": prop_type,
        "description": ""
    }
    
    if prop_type == "title":
        info["description"] = "Text (title)"
        info["python_type"] = "str"
        info["example"] = "Task Name"
    
    elif prop_type == "rich_text":
        info["description"] = "Rich text content"
        info["python_type"] = "str"
        info["example"] = "Description text"
    
    elif prop_type == "text":
        info["description"] = "Plain text"
        info["python_type"] = "str"
        info["example"] = "Text content"
    
    elif prop_type == "number":
        info["description"] = "Numeric value"
        info["python_type"] = "float | int"
        info["example"] = "42"
    
    elif prop_type == "select":
        options = prop_config.get("select", {}).get("options", [])
        option_names = [opt.get("name") for opt in options]
        info["description"] = "Single select dropdown"
        info["python_type"] = f"Literal[{', '.join(repr(opt) for opt in option_names)}]"
        info["possible_values"] = option_names
        if option_names:
            info["example"] = option_names[0]
    
    elif prop_type == "multi_select":
        options = prop_config.get("multi_select", {}).get("options", [])
        option_names = [opt.get("name") for opt in options]
        info["description"] = "Multiple select dropdown"
        info["python_type"] = f"list[Literal[{', '.join(repr(opt) for opt in option_names)}]]"
        info["possible_values"] = option_names
    
    elif prop_type == "date":
        info["description"] = "Date (ISO 8601 format)"
        info["python_type"] = "str (YYYY-MM-DD) | dict with 'start' and optional 'end'"
        info["format"] = "YYYY-MM-DD"
        info["example"] = datetime.now().strftime("%Y-%m-%d")
    
    elif prop_type == "checkbox":
        info["description"] = "Boolean checkbox"
        info["python_type"] = "bool"
        info["possible_values"] = [True, False]
    
    elif prop_type == "relation":
        info["description"] = "Relation to another database"
        info["python_type"] = "list[str]"
        info["example"] = "['related_page_id1', 'related_page_id2']"
    
    elif prop_type == "people":
        info["description"] = "Person/User"
        info["python_type"] = "list[dict] with 'id', 'object', 'person'"
        info["example"] = "[{id: '...', object: 'user', person: {...}}]"
    
    elif prop_type == "files":
        info["description"] = "File attachments"
        info["python_type"] = "list[dict] with file metadata"
        info["example"] = "[{name: 'file1.pdf', size: 1024, url: 'https://example.com/file1.pdf'}]"
    
    elif prop_type == "url":
        info["description"] = "URL link"
        info["python_type"] = "str"
        info["example"] = "https://example.com"
    
    elif prop_type == "email":
        info["description"] = "Email address"
        info["python_type"] = "str"
        info["example"] = "user@example.com"
    
    elif prop_type == "phone_number":
        info["description"] = "Phone number"
        info["python_type"] = "str"
        info["example"] = "+1-555-0100"
    
    elif prop_type == "formula":
        formula_config = prop_config.get("formula", {})
        result_type = formula_config.get("expression", "unknown")
        info["description"] = f"Formula result"
        info["formula_expression"] = result_type
        info["python_type"] = "str | number | bool (depends on formula)"
    
    elif prop_type == "rollup":
        rollup_config = prop_config.get("rollup", {})
        info["description"] = "Rollup aggregation"
        info["function"] = rollup_config.get("function", "unknown")
        info["python_type"] = "str | number | list (depends on aggregation)"
    
    elif prop_type == "created_time":
        info["description"] = "Page creation timestamp"
        info["python_type"] = "str (ISO 8601)"
        info["read_only"] = True
    
    elif prop_type == "created_by":
        info["description"] = "User who created the page"
        info["python_type"] = "dict with user info"
        info["read_only"] = True
    
    elif prop_type == "last_edited_time":
        info["description"] = "Last edit timestamp"
        info["python_type"] = "str (ISO 8601)"
        info["read_only"] = True
    
    elif prop_type == "last_edited_by":
        info["description"] = "User who last edited"
        info["python_type"] = "dict with user info"
        info["read_only"] = True
    
    else:
        info["description"] = f"Unknown type: {prop_type}"
        info["python_type"] = "Any"
    
    return info


def format_comprehensive_report(db_name: str, schema: Dict[str, Any]) -> str:
    """Format schema information into a comprehensive, human-readable report"""
    properties = schema.get("properties", {})
    property_infos = [extract_property_info(name, config) for name, config in properties.items()]
    
    report = []
    report.append("=" * 80)
    report.append(f"DATABASE: {db_name}")
    report.append("=" * 80)
    report.append("")
    
    report.append("PROPERTY DETAILS")
    report.append("-" * 80)
    for info in property_infos:
        report.append(f"\n{info['name']}")
        report.append(f"  Type: {info['type']}")
        report.append(f"  Python Type: {info['python_type']}")
        report.append(f"  Description: {info['description']}")
        
        if "possible_values" in info:
            report.append(f"  Possible Values: {info['possible_values']}")
        if "example" in info:
            report.append(f"  Example: {info['example']}")
        if info.get("read_only"):
            report.append(f"  ⚠️  READ-ONLY")
    
    report.append("")
    report.append("-" * 80)
    report.append("PYTHON TYPE HINTS")
    report.append("-" * 80)
    report.append("")
    report.append("from typing import Literal, TypedDict, Optional")
    report.append("")
    
    class_name = db_name.title().replace(" ", "").replace("-", "")
    report.append(f"class {class_name}Properties(TypedDict, total=False):")
    
    for info in property_infos:
        python_type = info["python_type"]
        report.append(f"    {info['name']}: Optional[{python_type}]")
    
    report.append("")
    return "\n".join(report)


def format_token_efficient_report(db_name: str, schema: Dict[str, Any]) -> str:
    """Format schema information in a concise, token-efficient way"""
    properties = schema.get("properties", {})
    property_infos = [extract_property_info(name, config) for name, config in properties.items()]
    
    report = []
    report.append(f"## {db_name}")
    report.append("")
    
    # Concise property list
    for info in property_infos:
        python_type = info["python_type"]
        name = info["name"]
        prop_type = info["type"]
        
        if "possible_values" in info:
            values = ", ".join(repr(v) for v in info["possible_values"])
            report.append(f"- `{name}`: {python_type} [{values}]")
        else:
            report.append(f"- `{name}`: {python_type}")
    
    report.append("")
    return "\n".join(report)


def main():
    print("\n🔍 Retrieving Notion Database Schemas...\n")
    
    # Fetch projects database
    print("Fetching Projects Database...")
    projects_schema = get_database_schema(PROJECTS_DB_ID)
    
    # Fetch tasks database
    print("Fetching Tasks Database...")
    tasks_schema = get_database_schema(TASKS_DB_ID)
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE REPORT (Human-Readable with Descriptions)")
    print("=" * 80)
    comprehensive_report = format_comprehensive_report("Projects Database", projects_schema)
    comprehensive_report += format_comprehensive_report("Tasks Database", tasks_schema)
    print(comprehensive_report)
    
    # Generate token-efficient report
    print("=" * 80)
    print("TOKEN-EFFICIENT REPORT (Concise)")
    print("=" * 80)
    token_efficient_report = format_token_efficient_report("Projects Database", projects_schema)
    token_efficient_report += format_token_efficient_report("Tasks Database", tasks_schema)
    print(token_efficient_report)
    
    # Save comprehensive report
    comprehensive_file = Path("data/context/database_schema_report_comprehensive.txt").absolute()
    with open(comprehensive_file, "w") as f:
        f.write(comprehensive_report)
    
    # Save token-efficient report
    token_file = Path("data/context/database_schema_report_token_efficient.txt").absolute()
    with open(token_file, "w") as f:
        f.write(token_efficient_report)
    
    print(f"\n✅ Comprehensive report saved to {comprehensive_file}")
    print(f"✅ Token-efficient report saved to {token_file}")
    print(f"\n📋 Copy the report of your choice and paste it into your LLM context!")


if __name__ == "__main__":
    main()
