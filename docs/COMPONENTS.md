# Components

This document explains crucial components, why they exist, and the main tradeoffs behind each one.

## 1) UI Bridge

Source: [src/presentation/ui_bridge.py](../src/presentation/ui_bridge.py)

Purpose:

- Hold small shared runtime signals for the CLI and graph streaming loop.
- Keep UI state outside core graph state.

Key fields:

- `current_node` for live progress display.
- `trial_num` for retry progress.
- `disambiguator` callback for interactive title resolution.
- `page_caches` for task-scoped cache handoff.

Tradeoff:

- This is global process state, which is simple and fast for a CLI process, but should be replaced by stronger isolation if moving to multi-tenant server runtime.

## 2) Async Page Cache

Source: [src/utils/page_cache.py](../src/utils/page_cache.py)

Purpose:

- Overlap fetch latency with ongoing graph execution.
- Deduplicate page fetches in a single run.

Design:

- Cache stores asyncio tasks keyed by normalized IDs.
- Prefetch can be triggered after resource resolution and execution.
- Supports refresh for mutated pages.

Tradeoff:

- Not serializable by design, so it stays out of graph state and is injected via runnable config.

## 3) Telemetry Wrapper

Source: [src/utils/telemetry.py](../src/utils/telemetry.py)

Purpose:

- Track which Notion pages were read or mutated by generated code.

Design:

- Injects a request interceptor into generated code before execution.
- Extracts IDs from page/block endpoint traffic.
- Injects `RESOURCE_MAP` directly into code scope.
- Supports fallback file-based affected-ID retrieval.

Tradeoff:

- Monkey-patching is pragmatic but requires careful re-entrancy handling in reused interpreters.

## 4) Environment Layering

Sources:

- [notion_query/environment.py](../notion_query/environment.py)
- [src/evaluation/sandbox.py](../src/evaluation/sandbox.py)

Purpose:

- Separate stable secrets from ephemeral evaluation IDs.

Design:

- `.env` for stable credentials/config.
- `.env.sandbox` for generated test IDs and short-lived sandbox artifacts.
- Runtime can load both with sandbox values overriding overlaps.

Tradeoff:

- Slightly more setup complexity in exchange for safer evaluation operations and fewer accidental edits of base secrets.

## 5) Precheck Guards

Sources:

- [src/guards.py](../src/guards.py)
- [src/nodes.py](../src/nodes.py)

Purpose:

- Stop unsafe/out-of-scope prompts early.
- Extract required resource titles before code generation.

Design:

- General guard returns strict JSON: reasoning, scope flag, required resources.
- Security guard uses Llama Guard result.
- Pipeline joins both checks before continuing.

Tradeoff:

- Additional model calls add latency but significantly improve reliability and safety.

## 6) Resource Resolution and Disambiguation

Source: [src/nodes.py](../src/nodes.py)

Purpose:

- Resolve user-mentioned page titles into concrete Notion IDs.

Design:

- Searches Notion by title for each required resource.
- Uses interactive disambiguation callback when multiple matches exist.
- Fails explicitly when no disambiguator is available in non-interactive runs.

Tradeoff:

- Extra pre-execution API calls reduce downstream code hallucination and bad ID usage.

## 7) Sandbox Lifecycle

Sources:

- [src/nodes.py](../src/nodes.py)
- [src/utils/execution_utils.py](../src/utils/execution_utils.py)

Purpose:

- Execute generated code in isolated runtime by default.

Design:

- Prepare/connect sandbox, execute with timeout, then cleanup.
- Egress policy allows only `api.notion.com`.
- Sends only `NOTION_*` env values to sandbox.

Tradeoff:

- Slight startup overhead for sandbox setup, but improved isolation and safer execution.

## 8) Egress Security Scan

Source: [src/nodes.py](../src/nodes.py)

Purpose:

- Prevent accidental secret leakage in outputs.

Design:

- Post-execution scan checks output text for configured sensitive token values.
- If leaked, output is replaced and terminal status is set to `security_blocked`.

Tradeoff:

- String matching is conservative and simple; false positives are possible but acceptable for safety.

## 9) OpenAI Client Session Management

Source: [src/utils/openai_utils.py](../src/utils/openai_utils.py)

Purpose:

- Keep async LLM client lifecycle bounded to each run.

Design:

- Context-managed session with contextvar storage.
- Reset and close at end of lifecycle.

Tradeoff:

- Slight setup/teardown overhead per run, but avoids stale client/resource leaks across CLI turns.

## 10) Hardcoded Context Registry

Source: [src/models/hardcoded_contexts.py](../src/models/hardcoded_contexts.py)

Purpose:

- Provide deterministic retrieval context variants without requiring live vector retrieval.

Design:

- Loads context files from `data/context` plus in-code baselines.
- Builds named combinations (for example schema report + Notion API summary).
- Supports `dynamic` mode as a separate route.

Tradeoff:

- Strong determinism and simpler ops, but less adaptive than live retrieval.

## 11) Error Analysis Automation

Sources:

- [src/error_analysis.py](../src/error_analysis.py)
- [src/evaluation/shared.py](../src/evaluation/shared.py)

Purpose:

- Convert evaluation outputs into structured diagnostics and summaries.

Design:

- Can run automatically after evaluation orchestration.
- Supports section toggles for code, execution, statements, and retrieval context.

Tradeoff:

- Additional post-processing time, but much faster iteration on failure patterns.

## 12) State Fields Most Relevant to Users

Source: [src/models/schema.py](../src/models/schema.py)

Frequently surfaced fields:

- `execution_output`
- `message_to_user`
- `feedback`
- `verdict`
- `affected_notion_ids`
- `relevant_page_ids`
