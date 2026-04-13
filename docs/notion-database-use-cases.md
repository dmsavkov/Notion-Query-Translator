Mapping the absolute perimeter of user intent is exactly how you bulletproof an AI architecture. You are no longer just building a script; you are defining the **Domain Boundary** of your agent.

To guarantee your dual-layer IPC (Telemetry + JSON Envelope) handles everything without dropping data or crashing the UI, we must categorize the Notion database interactions into distinct vectors.

Here is the exhaustive intent matrix for a Notion database AI agent.

### Vector 1: The Mutators (Side-Effect Heavy)

These operations trigger your Telemetry wrapper and require UI hydration via the `AsyncPageCache`.

- **1.1 Single Property State Change:** _"Mark the 'Server Migration' task as done."_ (Standard `PATCH`).
- **1.2 Bulk Property State Change:** _"Move all tasks assigned to Olya to 'In Progress'."_ (Requires querying the DB first, then firing multiple `PATCH` requests).
- **1.3 Data Extraction & Entry:** _"Take these 5 unstructured meeting notes and create a task for each in the tracker."_ (Parsing text into structured `POST` requests).
- **1.4 The Deep Content Creator:** _"Create a project brief page in the DB and draft a 3-paragraph summary inside it."_ (Requires `POST` to the DB, followed by `PATCH` to the new page's block children. Telemetry must catch the parent UUID).
- **1.5 The Relational Linker:** _"Link the new bug report to the 'Backend Overhaul' epic."_ (Updating a `relation` property. This is highly complex for the LLM because it requires querying _two_ databases to get both UUIDs before patching).
- **1.6 The Archiver (Soft Delete):** _"Archive all tasks completed before January."_ (`PATCH` with `archived: true`).

### Vector 2: The Extractors (Hydration Heavy)

These operations bypass Telemetry completely. They rely 100% on the JSON Envelope returning `relevant_page_ids` so the UI can render the results.

- **2.1 The Semantic Filter:** _"Show me the highest priority bugs that are currently blocking deployment."_ (Agent queries the DB, applies logic, returns the UUIDs).
- **2.2 The Time-Bound Query:** _"What tasks are due this week?"_ (Agent constructs a Notion API date filter).
- **2.3 The Personal Query:** _"What is on my plate today?"_ (Agent filters by the user's Notion Person ID).
- **2.4 The Pagination Trap:** _"Show me every completed task."_ (If the DB has 500 tasks, the agent must handle Notion's `next_cursor` pagination limit of 100 items per request).

### Vector 3: The Analysts (Compute Heavy)

These operations bypass Telemetry AND bypass UI Hydration. They rely strictly on the `message_to_user` field in the JSON Envelope.

- **3.1 Aggregation & Math:** _"What is the total budget allocated to active projects?"_ (Agent fetches data, runs Python math, prints the string).
- **3.2 Synthesis & Summarization:** _"Read the descriptions of the last 3 closed bugs and give me a 1-sentence summary of what went wrong."_ (Agent fetches block content, uses its LLM context to summarize, prints the string).
- **3.3 The Schema Inspector:** _"What properties do I need to fill out to create a new employee record?"_ (Agent fetches the DB schema, lists the columns).

### Vector 4: The Agentic Workflows (Multi-Step Logic)

These are the highest-tier interactions where the LLM acts truly autonomously, blending vectors together.

- **4.1 Read -> Decide -> Mutate:** _"Find any tasks that have been 'In Progress' for more than 14 days and tag them with 'Stale'."_
- **4.2 The Cross-Database Sync:** _"For every new client in the CRM database, create a corresponding project template in the Projects database."_
- **4.3 The Self-Correction Workflow:** The user asks for something impossible (e.g., _"Set the status to 'Finished'"_ but the allowed options are only 'Done' or 'In Progress'). The agent hits a 400 Validation Error, catches the traceback, reads the valid options, and retries the mutation autonomously.

---

### The Architectural Confidence Check

Because of the design decisions you have made so far:

1.  **Vector 1** is perfectly handled by your Telemetry + Forced Refresh.
2.  **Vector 2** is perfectly handled by the JSON Envelope `relevant_page_ids`.
3.  **Vector 3** is perfectly handled by the JSON Envelope `message_to_user`.
4.  **Vector 4** is natively supported because the Sandbox is executing raw Python, allowing loops, conditional logic, and multiple API calls within a single turn.
