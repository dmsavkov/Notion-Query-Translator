# Architecture Review

## Overview

This document captures the key architectural decisions for the Notion Query Translator project.

<summary>Core Engine Design</summary>

The system uses a **LangGraph state machine** to orchestrate the full pipeline from query understanding to code execution.

### Key Components

1. **RAG Pipeline** — retrieves relevant Notion API docs
2. **Code Generator** — produces executable Python
3. **Sandbox Executor** — runs code safely in E2B

<callout icon="💡">Always validate the generated code against the egress security policy before returning results.</callout>

<page url="https://notion.so/some-page-id">Related Design Document</page>

<database url="https://notion.so/db-id">Task Tracker</database>

<mention-page url="https://notion.so/ref-page" title="Referenced"/>

### Data Flow

<table>
<tr><th>Step</th><th>Component</th><th>Output</th></tr>
<tr><td>1</td><td>Precheck</td><td>Safety verdict</td></tr>
<tr><td>2</td><td>Retrieve</td><td>API context</td></tr>
<tr><td>3</td><td>Codegen</td><td>Python script</td></tr>
<tr><td>4</td><td>Execute</td><td>API response</td></tr>
</table>

<details>
<summary>Implementation Notes</summary>

These are internal notes about the implementation that are collapsible.

</details>

<empty-block/>

> **Note:** This is a standard blockquote that should be preserved.

```python
def example():
    return "Hello from code block"
```
