# Implementation Notes

This document explains a few internal features that are easy to miss when reading the app from the outside.

## Page Caching

The app keeps a small in-memory cache of Notion pages during a single run. The cache lives in [src/utils/page_cache.py](../src/utils/page_cache.py) and is injected through `RunnableConfig.configurable`.

What it does:

- When the graph resolves a page or block ID, it can start fetching the page properties and markdown in the background.
- The cache deduplicates requests, so the same page is not fetched twice in the same turn.
- The CLI harvests the prefetched pages after execution and passes them into the viewer, so the final display can reuse already-fetched data.

Why it exists:

- It hides some of the latency from the user by overlapping fetch work with the rest of the pipeline.
- It reduces duplicate Notion API calls when multiple parts of the pipeline need the same page.
- It keeps the UI responsive without moving the actual execution logic into the presentation layer.

Important detail:

- The cache is intentionally not stored in LangGraph state because it contains asyncio tasks and is not serializable.
- It is a turn-scoped helper, not a long-term application cache.

## Async Client Closing

The OpenAI-compatible async client is created inside a request-scoped context manager in [src/utils/openai_utils.py](../src/utils/openai_utils.py).

What it does:

- `openai_client_session()` creates the async client at the start of a pipeline run.
- The client is stored in a context variable so lower-level helpers like `async_chat_wrapper()` can retrieve it without threading the client through every function.
- When the run finishes, the context manager resets the context variable and explicitly closes the client.

Why it exists:

- The CLI can execute multiple prompts in the same process, so clients must not leak across turns.
- Explicit closing prevents stale sockets and shutdown errors when the event loop is torn down.
- It keeps the OpenAI lifecycle aligned with the pipeline lifecycle instead of relying on module globals.

Practical consequence:

- Each run gets a fresh client context.
- If the client is not closed, the next CLI turn can hit `Event loop is closed` during teardown.

## Telemetry Patching

Generated code is wrapped with a lightweight telemetry shim in [src/utils/telemetry.py](../src/utils/telemetry.py).

What it does:

- It monkey-patches `requests.Session.request` inside the generated code before execution starts.
- The patch watches requests to Notion page and block endpoints.
- It records page IDs that were read and page IDs that were mutated.
- After execution, it writes those IDs to a JSON file so the host app can read them back.

Why it exists:

- The generated script often does not print a clean summary of which pages it touched.
- Telemetry gives the app a reliable post-run signal for rendering affected pages in the viewer.
- Separating `read` and `mutated` IDs makes it possible to explain both what was inspected and what was changed.

Why it needs care:

- The sandbox can reuse a Python interpreter, so the patch must be re-entrant.
- The current wrapper preserves the original request callable on the patched function so later runs do not recurse into an already-wrapped request method.

## Title Extraction

The app uses title extraction in two places: disambiguation and rendering.

Disambiguation:

- In [src/nodes.py](../src/nodes.py), `_extract_result_title()` reads the title property from Notion search results.
- It is used when multiple pages match the same user-entered title and the CLI needs to show human-friendly choices.

Rendering:

- In [src/presentation/viewer.py](../src/presentation/viewer.py), `_extract_page_title()` tries common property names such as `Name`, `Title`, `title`, and `name`.
- If none of those are present, it falls back to a short `Page <id>` label.

Why it exists:

- Notion search results and page objects do not always share the same schema shape.
- Humans need readable labels during disambiguation and when the final page view is rendered.
- The fallback keeps the UI usable even when a page has an unusual schema or no obvious title field.

What it is not:

- It is not a canonical schema resolver.
- It is a best-effort UI heuristic so the app can present useful names without failing when title fields vary.

## Output Fields You Will See

The application currently surfaces these user-visible fields most often:

- `execution_output`: main text shown for success or failure output.
- `feedback`: extra explanation for failed executions and reflection loops.
- `verdict`: structured evaluation or reflection result.
- `affected_notion_ids`: the pages the telemetry layer says were touched.

There is no separate `message_to_user` field in the current state schema.
