# Evaluation System

This document describes the evaluation entry points in this repository, how they are wired together, and the main design choices and caveats that matter when changing them.

## Shared Orchestration

The evaluation scripts live under [evaluation/](../evaluation) and are all built on the same shared path:

1. A script defines an evaluation settings object, usually based on [StandardEvaluationSettings](../src/evaluation/utils.py).
2. The script loads task specs from `evals/`.
3. [evaluation_orchestration](../src/evaluation/shared.py) ensures the LangSmith dataset exists, refreshes examples if inputs drift, and then calls `aevaluate(...)`.
4. The target function returns a structured output dict.
5. One or more evaluators compare that output against the dataset reference outputs.

The orchestration layer is intentionally narrow: it raises on runtime execution errors, not on low metric scores. A bad prediction should show up as a score or comment, while an actual crash should fail the run.

The important design decision is that the dataset is the canonical source of truth for prompts and reference outputs. The eval scripts should stay thin and should not re-encode task truth in code unless they are deliberately deriving it.

## General Precheck Evaluation

Entry point: [evaluation/test_general_precheck.py](../evaluation/test_general_precheck.py)

This evaluation checks the first-entry guardrail that classifies a user request before any resolution work happens.

Flow:

1. The target builds node state with [build_node_eval_state](../src/evaluation/utils.py).
2. It calls [precheck_general_node](../src/nodes.py).
3. The guardrail prompt in [src/guards.py](../src/guards.py) asks the model to return strict JSON with `reasoning`, `relevant_to_notion_scope`, and `required_resources`.
4. The target normalizes the returned `required_resources` list.
5. Evaluators compare the predicted values with the reference outputs from the dataset.

Metrics:

- `relevant_to_notion_scope_match`: exact match on the boolean flag.
- `required_resources_match`: normalized list equality against the reference titles.

Design choice:

- This evaluation measures inference quality, not Notion lookup quality. A correct precheck can still fail later if the resolver cannot find or disambiguate the pages.

## Title Search Evaluation

Entry point: [evaluation/test_title_search.py](../evaluation/test_title_search.py)

This evaluation measures the flow from general precheck through Notion title resolution.

Flow:

1. The target builds node state from the sample input.
2. It runs [precheck_general_node](../src/nodes.py) first.
3. The returned `meta.required_resources` is normalized and written back into `state.meta.required_resources`.
4. The target then calls [resolve_resources_node](../src/nodes.py), which performs the Notion search and resolves page IDs.
5. The target captures the raw search observations so the evaluator can inspect what the search returned.
6. Evaluators compare the observed search behavior and the resolved count against the reference titles in the dataset.

Metrics:

- `top1_precision`: for each expected title from `reference_outputs.required_resources`, the metric checks whether any query returned that title as the first search result after title normalization.
- `top3_recall`: the same check, but across the first three search results.
- `precheck_mention_count_match`: compares the number of expected titles with the number of pages mentioned by the general precheck. The resolved page count is still included in the comment for debugging, but it does not affect the score.

Important caveats:

- `required_resources` is intentionally not duplicated in the title-search input state. The expected titles live in `reference_outputs`, while the live state is derived from the precheck step.
- Title matching is normalized with lowercase, underscore-to-space conversion, and whitespace collapsing. The evaluation does not depend on exact capitalization.
- The search is still a fuzzy Notion search, not an exact title lookup. Ambiguity handling depends on the resolver and any configured disambiguator.
- The names `top1_precision` and `top3_recall` are descriptive, but they are really per-reference-title hit-rate checks, not full corpus-level retrieval metrics.

Design choice:

- This evaluation intentionally validates the handoff between precheck and resolver. It is not useful to treat the sample input as the source of truth for the expected titles, because that would hide the inference step we are trying to test.

## Codegen + Reflect Evaluation

Entry point: [evaluation/test_code_generation.py](../evaluation/test_code_generation.py)

This evaluation runs a partial live pipeline rather than a single node.

Flow:

1. The target builds node state and hydrates required titles into a `resource_map`.
2. Title hydration uses Notion search plus direct page fetch validation before execution begins.
3. The target loads a static retrieval context when needed and builds a `general_info` block.
4. The pipeline resumes from the configured starting node and runs until the configured interrupt point.
5. The evaluation then scores the generated result with separate code-quality and execution-quality evaluators.

Metrics:

- `code_statements_score`: judge-model scoring of whether the generated code contains the required statements.
- `code_execution_score`: pass/fail scoring of the execution result.

Important caveats:

- This evaluation is more expensive and more variable than the node-level checks because it depends on a judge model, sandbox execution, and partial graph behavior.
- Required titles can come from the sample input first, with task-spec fallback if needed.
- The hydration step validates resolved IDs with a direct GET call, so stale or invalid page IDs fail early.

## Shared Design Decisions

The evaluation stack follows a few recurring rules:

- Keep orchestration shared and thin. The scripts should define intent and metrics, while [src/evaluation/shared.py](../src/evaluation/shared.py) owns the LangSmith plumbing.
- Keep dataset examples up to date. [ensure_dataset](../src/evaluation/utils.py) updates existing LangSmith examples when task inputs change, so YAML edits propagate to already-created datasets.
- Keep title handling consistent. Normalization and extraction helpers should come from the shared Notion request layer rather than being reimplemented in each eval.
- Keep runtime state separate from reference truth. If a metric is about inference, the reference should live in the dataset outputs, not in the input payload.

## Practical Caveats

- General precheck and title search are not measuring the same thing. One measures classification and resource inference; the other measures search and resolution quality.
- Search ranking can vary, so small changes in Notion data can change top1/top3 scores without changing the underlying precheck quality.
- Existing LangSmith examples can become stale when YAML changes. If a task gains or loses `required_resources`, the dataset should be refreshed through the normal orchestration path.
- The title-search target depends on a precheck model call, so tests should mock that node when they need deterministic behavior.

## Related Notes

If you want implementation details for the runtime app rather than the eval layer, see [Implementation Notes](implementation-notes.md).
