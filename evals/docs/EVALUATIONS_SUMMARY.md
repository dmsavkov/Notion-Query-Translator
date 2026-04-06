# Notion API Evaluations - Comprehensive Summary

This document provides a complete overview of all evaluation tasks for the Notion Query project, organized by complexity level. Each evaluation tests specific Notion API capabilities and patterns.

---

## SIMPLE EVALUATIONS

Simple evaluations test single, focused operations or straightforward API calls with minimal branching logic. These form the foundation for understanding basic Notion API patterns.

### 1. **Add Task** (`add_task.yaml`)

**Objective:** Create a new task in the Notion Tasks database with multiple properties.

**Task Details:**

- Title: "Add_task_test"
- Due Date: Today (dynamically computed)
- Importance: 4
- Project ID: Retrieved from environment variable

**API Operations:**

- Single POST request to `https://api.notion.com/v1/pages`
- Parent: Tasks database ID
- Properties set: Name (title), Due Date (date), Importance (select), Project (relation)

**Key Concepts:**

- Dynamic date parsing: "today" → ISO 8601 format (YYYY-MM-DD)
- Property type handling: title, date, select, relation
- Environment variable substitution for project reference
- Request headers: Authorization, Content-Type, Notion-Version (2022-06-28)

**Correct Statements:**

- Endpoint is POST https://api.notion.com/v1/pages for creating database pages
- Date values must be in ISO 8601 format (YYYY-MM-DD)
- Property names must exactly match Notion database schema
- The "Title" property uses type "title" with rich_text array wrapper

---

### 2. **Add Toggle to Page** (`add_toggle_to_page.yaml`)

**Objective:** Add a toggle block to a page with nested paragraph content.

**Task Details:**

- Target Page ID: 24ecb17dcc4480beb3a6e6f0e4751989
- Toggle Title: "Add_toggle_test"
- Toggle Content: "This is a test toggle"

**API Operations:**

- Single PATCH request to `https://api.notion.com/v1/blocks/{page_id}/children`
- Block type: toggle with nested children
- Nested content: paragraph block inside toggle

**Key Concepts:**

- Block structure: object, type, and type-specific properties
- Toggle composition: header text + children array with nested blocks
- Proper nesting of block objects within toggle.children
- Rich text formatting for both toggle header and paragraph content

**Correct Statements:**

- Endpoint is PATCH https://api.notion.com/v1/blocks/{page_id}/children for appending blocks
- Toggle blocks support a "children" array with nested blocks
- Each block must have "object": "block" and appropriate "type" field
- Rich text appears in both toggle header and nested paragraph content

---

### 3. **Append Checklist to Page** (`append_checklist_to_page_synthetic.yaml`)

**Objective:** Add multiple checklist items to a page.

**Task Details:**

- Target: Inbox page (from environment variable)
- Checklist Items: "Read documentation", "Draft code", "Test API"

**API Operations:**

- Single PATCH request to `https://api.notion.com/v1/blocks/{page_id}/children`
- Block type: to_do (unchecked by default)
- Dynamic block generation from item array

**Key Concepts:**

- Batch appending: send all checklist items in a single "children" array
- To-do block structure: rich_text array + checked boolean
- Iteration pattern: convert item list into block objects
- Efficient payload design: one request for multiple items

**Correct Statements:**

- Endpoint is PATCH https://api.notion.com/v1/blocks/{page_id}/children
- Multiple blocks can be appended in a single request via the "children" array
- To-do blocks use type "to_do" with rich_text and checked properties
- Checked defaults to false; set to true for pre-checked items

---

### 4. **Get Page Content** (`get_page_content_synthetic.yaml`)

**Objective:** Retrieve all blocks/content from a page with proper pagination.

**Task Details:**

- Target: Inbox page
- Page Size: 100 (max per request)
- Pagination: Handle multi-page results

**API Operations:**

- GET request to `https://api.notion.com/v1/blocks/{page_id}/children`
- Query parameter: page_size=100
- Pagination handling: next_cursor and has_more for multiple pages

**Key Concepts:**

- Block retrieval API vs. page properties API
- Pagination loop pattern: fetch until has_more=false or next_cursor=null
- Response structure: results array containing block objects
- Block parsing: each block has type and type-specific nested object

**Correct Statements:**

- Endpoint is GET https://api.notion.com/v1/blocks/{page_id}/children
- Max page_size is 100; larger result sets require pagination loop
- Response results are in a "results" key as an array of block objects
- Each block has a "type" field (paragraph, heading_1, to_do, etc.) and matching nested content key
- For subsequent pages, pass previous response's "next_cursor" as "start_cursor" in next request

---

### 5. **Retrieve Sorted Tasks** (`retrieve_tasks.yaml`)

**Objective:** Query the Tasks database with multi-field sorting.

**Task Details:**

- Database: Tasks
- Limit: 3 results
- Sort by: Importance (descending), then Urgency (descending)

**API Operations:**

- Single POST request to `https://api.notion.com/v1/databases/{database_id}/query`
- page_size: 3
- sorts array: multiple sort criteria with property names and directions

**Key Concepts:**

- Database query endpoint vs. page creation
- Multi-level sorting: primary and secondary sort fields
- Direction values: "ascending" or "descending"
- Result limiting via page_size parameter

**Correct Statements:**

- Endpoint is POST https://api.notion.com/v1/databases/{database_id}/query
- Result count is limited via "page_size" in request body
- Multiple sort criteria are defined in a "sorts" array
- Each sort entry requires "property" (exact column name) and "direction"
- Sorts are applied in order: earlier entries are primary, later entries are tiebreakers

---

### 6. **Update Task Status** (`update_task_status_synthetic.yaml`)

**Objective:** Modify page properties: update status and clear due date.

**Task Details:**

- Target Page ID: From NOTION_ID_UPDATE_PAGE_ID environment variable
- New Status: "Done" (select property)
- Due Date Action: Clear (set to null)

**API Operations:**

- Single PATCH request to `https://api.notion.com/v1/pages/{page_id}`
- Top-level properties object
- Two property updates: Status (select) and Due Date (date with null)

**Key Concepts:**

- Page PATCH endpoint vs. database query
- Select property update: must provide property name and select value
- Date clearing: set date.start to None/null
- Batch property updates in single request

**Correct Statements:**

- Endpoint is PATCH https://api.notion.com/v1/pages/{page_id}
- Property names must exactly match your Notion database schema
- Status updates use "select" type with {"name": "status_value"} structure
- Dates are cleared by setting the value to None (not deleting the property)
- Multiple properties can be updated in a single PATCH request

---

## COMPLEX EVALUATIONS

Complex evaluations combine multiple API calls, conditional logic, and advanced patterns. They demonstrate real-world workflows requiring planning and multi-step execution.

### 1. **Archive Stale Tasks** (`complex/archive_stale_tasks.yaml`)

**Objective:** Find old tasks and archive them with comments.

**Task Details:**

- Threshold: Tasks with "Last Reviewed" date older than 30 days
- Actions: Archive each task AND add a comment

**Multi-Step Workflow:**

1. **Query Phase:** POST to database query endpoint
   - Filter: Date property "Last Reviewed" on_or_before (calculated date)
   - Result: List of matching task page IDs

2. **Archive Phase:** For each matched page
   - PATCH https://api.notion.com/v1/pages/{page_id}
   - Set top-level "archived": true

3. **Comment Phase:** For each matched page
   - POST https://api.notion.com/v1/comments
   - Parent: page_id reference
   - Content: Rich text message "This task is stale. Archived."

**Key Concepts:**

- Date arithmetic: timedelta(days=30) from current date
- Date filter operator: "on_or_before" for threshold-based selection
- Two-action pattern per page: archive then comment
- Comment API payload structure with parent reference
- Loop structure for bulk operations with multiple requests per item

**Advanced Patterns:**

- Temporal filtering (computed thresholds)
- Multi-operation workflows per resource
- Comments as audit trail / side effect
- Computed data passed between request types

**Correct Statements:**

- Python datetime arithmetic computes threshold as current date minus timedelta(days=30)
- Query filter shape: {"property": "Last Reviewed", "date": {"on_or_before": "<date>"}}
- Archive uses PATCH https://api.notion.com/v1/pages/{page_id} with {"archived": true}
- Comments use POST https://api.notion.com/v1/comments with parent.page_id reference
- Both archive and comment are performed per page (two requests per page, not one global request)

---

### 2. **Create Task with Blocks and Relation** (`complex/create_task_with_blocks_and_relation.yaml`)

**Objective:** Create a task with checklist items and a relation to a project.

**Task Details:**

- Task Title: "Architecture Review"
- Project Name: "AI Research" (requires lookup)
- Nested Checklist: "Review Postgres", "Review MongoDB"

**Multi-Step Workflow:**

1. **Project Lookup Phase:** POST to projects database query
   - Filter: Name equals "AI Research"
   - Result: Project page ID for relation

2. **Task Creation Phase:** POST to create new page
   - Parent: Tasks database
   - Properties: Name (title), Project (relation with resolved ID)
   - Nested Blocks: Two to_do blocks in children array

**Key Concepts:**

- Forward reference resolution: lookup related resource before main creation
- Relation property: reference resolved page ID, not name string
- Atomic page creation: properties + nested blocks in single POST
- Children array for initial block structure at creation time
- Property type handling: title, relation with ID array

**Advanced Patterns:**

- Dependency resolution (finding project before creating task)
- Database queries for validation and reference lookup
- Relation properties with ID resolution
- Nested content creation in single request vs. sequential appends

**Correct Statements:**

- First request queries projects database to resolve name to ID
- Project lookup uses title filter: {"property": "Name", "title": {"equals": "AI Research"}}
- Task creation is POST https://api.notion.com/v1/pages
- Relation property uses {"relation": [{"id": "<resolved_project_id>"}]}
- Nested blocks are defined in "children" array at creation time
- To_do blocks can be added as initial children during page creation
- Lookup failure handling: return with "project_not_found" reason

---

### 3. **Propagate Status Done** (`complex/propagate_status_done.yaml`)

**Objective:** Cascade "Done" status through blocked task relationships.

**Task Details:**

- Target Task: "New problem" (lookup by name)
- Condition: Only propagate if Status is already "Done"
- Action: Mark all blocked tasks as "Done"

**Multi-Step Workflow:**

1. **Lookup Phase:** Query tasks database by name
   - Filter: Name equals "New problem"
   - Result: Task page ID

2. **Status Check Phase:** GET single page details
   - Fetch full page properties
   - Extract Status select value
   - Early return if status ≠ "Done"

3. **Relation Read Phase:** Extract from page properties
   - Read "Blocking" relation property (array of page references)
   - Convert to flat list of blocked task IDs

4. **Propagation Phase:** For each blocked task
   - PATCH each task with Status = "Done"
   - Request per blocked task

**Key Concepts:**

- Conditional execution based on property value check
- Relation properties as dependencies (blocking relationships)
- Early termination for unsatisfied conditions
- Bulk updates triggered by single source state change
- Property extraction from complex response structure

**Advanced Patterns:**

- Conditional workflow: gate propagation on parent state
- Relation navigation: extract references from relation arrays
- State-based triggers (only propagate when parent is done)
- Deep property access: properties.Status.select.name navigation

**Correct Statements:**

- Task lookup uses title filter: {"property": "Name", "title": {"equals": "New problem"}}
- Result constraint page_size: 1 optimizes single-item lookup
- Status is accessed via properties.Status.select.name
- Propagation occurs only when status_name == "Done"
- Blocked relations are extracted from properties.Blocking.relation as array
- Each blocked task is updated independently with PATCH https://api.notion.com/v1/pages/{blocked_id}
- PATCH payload: {"properties": {"Status": {"select": {"name": "Done"}}}}

---

### 4. **Update Imminent Tasks** (`complex/update_imminent_tasks.yaml`)

**Objective:** Find tasks due in exactly N days and boost their intensity.

**Task Details:**

- Time Window: Exactly 3 days from today
- Target Property: Due Date
- Update: Intensity = "8"

**Multi-Step Workflow:**

1. **Date Calculation Phase:**
   - Compute target date: today + timedelta(days=3)
   - Format as ISO 8601 string

2. **Query Phase:** POST database query
   - Filter: Due Date equals (calculated date)
   - Result: List of matching task IDs

3. **Update Phase:** For each matched task
   - PATCH with Intensity select value "8"

**Key Concepts:**

- Temporal computation for exact date matching
- Date filter operator: "equals" for precise matching on single day
- Select property update with string value
- Loop-based batch updates without explicitly requesting all pages

**Advanced Patterns:**

- Time-sensitive queries (due dates in future window)
- Computed filter values from temporal arithmetic
- Efficient batch filtering followed by sequential updates
- Date format consistency (ISO 8601 throughout)

**Correct Statements:**

- Target date computed as current date plus timedelta(days=3)
- Date converted to ISO format 'YYYY-MM-DD' before query
- Query filter: {"property": "Due Date", "date": {"equals": "<calculated_date>"}}
- Each matched page updated via PATCH https://api.notion.com/v1/pages/{page_id}
- Intensity update: {"properties": {"Intensity": {"select": {"name": "8"}}}}

---

### 5. **Vague Emergency Escalation** (`complex/vague_emergency_escalation.yaml`)

**Objective:** Handle ambiguous natural language request by inferring schema mappings.

**Task Details:**

- Natural Language: "Find tasks that are completely unstarted but extremely urgent, and mark them to execute right now"
- Semantic Mappings Performed:
  - "completely unstarted" → Status = "Not started"
  - "extremely urgent" → Urgency = "4"
  - "execute right now" → Do Now checkbox = true

**Multi-Step Workflow:**

1. **Semantic Resolution Phase:** Map vague language to schema values
   - Unstarted → Status exact value "Not started"
   - Extremely urgent → Urgency exact value "4"
   - Execute right now → checkbox property "Do Now"

2. **Query Phase:** POST database query with compound filter
   - AND condition with Status and Urgency predicates
   - Result: List of matching task IDs

3. **Update Phase:** For each matched task
   - PATCH with Do Now checkbox = true

**Compound Filter Structure:**

```
{
  "and": [
    {"property": "Status", "select": {"equals": "Not started"}},
    {"property": "Urgency", "select": {"equals": "4"}}
  ]
}
```

**Key Concepts:**

- Natural language interpretation into structured schema
- Heuristic mappings from vague terms to precise values
- Compound filters: AND/OR logic for multi-condition queries
- Multiple property types in single query (select + checkbox)
- Inference-based property selection

**Advanced Patterns:**

- Semantic understanding: mapping user intent to data schema
- Compound filtering: combining multiple predicates with logic operators
- Checkbox property type distinct from select type
- Heuristic approach: reasoning about likely property values

**Correct Statements:**

- Semantic mappings: "unstarted" → Status "Not started", "extremely urgent" → Urgency "4", "execute right now" → Do Now checkbox
- Query uses compound filter structure with "and" array
- Status predicate: {"property": "Status", "select": {"equals": "Not started"}}
- Urgency predicate: {"property": "Urgency", "select": {"equals": "4"}}
- Pages updated via PATCH https://api.notion.com/v1/pages/{page_id}
- Update payload: {"properties": {"Do Now": {"checkbox": true}}}

---

## CROSS-CUTTING PATTERNS & REFERENCE

### API Endpoints Summary

| Operation      | Method | Endpoint                            | Use Case                       |
| -------------- | ------ | ----------------------------------- | ------------------------------ |
| Create Page    | POST   | `/v1/pages`                         | Add new database entry or page |
| Update Page    | PATCH  | `/v1/pages/{page_id}`               | Modify properties, archive     |
| Get Page       | GET    | `/v1/pages/{page_id}`               | Fetch single page details      |
| Append Blocks  | PATCH  | `/v1/blocks/{page_id}/children`     | Add content to page            |
| Get Blocks     | GET    | `/v1/blocks/{page_id}/children`     | Retrieve page content          |
| Query Database | POST   | `/v1/databases/{database_id}/query` | Filter and sort pages          |
| Add Comment    | POST   | `/v1/comments`                      | Post comment on page           |

### Property Types Reference

| Type      | Field Name | Structure                                        | Example              |
| --------- | ---------- | ------------------------------------------------ | -------------------- |
| Title     | `title`    | `[{"type": "text", "text": {"content": "..."}}]` | Page name            |
| Select    | `select`   | `{"name": "value"}`                              | Status, Category     |
| Date      | `date`     | `{"start": "YYYY-MM-DD", "end": "..."}`          | Due dates            |
| Relation  | `relation` | `[{"id": "..."}]`                                | Links to other pages |
| Checkbox  | `checkbox` | `true/false`                                     | Boolean flags        |
| Rich Text | (various)  | `[{"type": "text", "text": {...}}]`              | Text content         |

### Block Types Reference

| Type                 | Use Case            | Key Properties          |
| -------------------- | ------------------- | ----------------------- |
| `paragraph`          | Text content        | `rich_text`             |
| `to_do`              | Checklist item      | `rich_text`, `checked`  |
| `toggle`             | Collapsible section | `rich_text`, `children` |
| `heading_1/2/3`      | Headings            | `rich_text`             |
| `bulleted_list_item` | Bullet point        | `rich_text`             |

### Common Filter Operators

| Operator       | Applies To          | Example             |
| -------------- | ------------------- | ------------------- |
| `equals`       | Select, Title, Date | Exact match         |
| `on_or_before` | Date                | Threshold filtering |
| `equals`       | Date                | Single day matching |
| And/Or         | Any                 | Compound conditions |

### Headers Required

All Notion API requests require:

```
Authorization: Bearer {NOTION_TOKEN}
Content-Type: application/json
Notion-Version: 2022-06-28
```

---

## LEARNING PROGRESSION

**Foundation (Simple Evals 1-6):**

- Basic CRUD operations on pages
- Property type handling
- Single-request workflows
- Date formatting and pagination

**Intermediate (Complex Evals 1-5):**

- Multi-step workflows with computed values
- Query-based filtering and sorting
- Conditional execution and early returns
- Relation properties and resolution
- Batch operations with loops

**Advanced Conceptual Patterns:**

- Semantic interpretation of vague requirements
- Temporal querying and arithmetic
- Audit trails through side effects (comments)
- Graph-like propagation through relations
- Compound boolean logic in filters

---

## TESTING & VERIFICATION NOTES

- Each evaluation includes a `correct_statements` section documenting precise API behaviors
- Complex evals demonstrate patterns that appear across real-world Notion automation tasks
- Simple evals provide building blocks; complex evals show composition into meaningful workflows
- Natural language mapping (vague_emergency_escalation) represents AI agent reasoning challenges
- Temporal operations (archive_stale, update_imminent) test date computation accuracy
- Relation navigation (propagate_status, create_with_relation) test graph traversal patterns
