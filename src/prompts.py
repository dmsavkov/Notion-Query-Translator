"""
Central store for all LLM prompt templates and builder functions.

Static constants:    uppercase names, imported directly.
Builder functions:   build_*() — accept runtime parameters, return the full prompt string.
"""

from typing import Dict, List, Optional


# ── Static Prompt Constants ────────────────────────────────────────────────────

CODEGEN_ENV_CONTEXT = (
    "<notion_env_keys>\n"
    "Available environment variables:\n"
    "  NOTION_TOKEN                – API bearer token\n"
    "  NOTION_TASKS_DATABASE_ID    – Database ID for tasks\n"
    "  NOTION_TASKS_DATA_SOURCE_ID – Data source ID for tasks\n"
    "  NOTION_PROJECTS_DATABASE_ID – Database ID for projects\n"
    "  NOTION_PROJECTS_DATA_SOURCE_ID – Data source ID for projects\n"
    "  NOTION_INBOX_PAGE_ID        – Target page ID for additions (for appending, inserting, updating content)\n"
    "</notion_env_keys>"
)

GENERATE_TESTS_KEYS_CONTEXT = (
    "<notion_env_keys>\n"
    "CRITICAL: The solution module reads ALL secrets via os.getenv().\n"
    "Your tests MUST patch env vars — never construct a NotionKeys object or pass real secrets.\n\n"
    "  NOTION_TOKEN                – Notion API bearer token\n"
    "  NOTION_TASKS_DATABASE_ID    – Tasks database identifier\n"
    "  NOTION_TASKS_DATA_SOURCE_ID – Tasks data-source identifier\n"
    "  NOTION_PROJECTS_DATABASE_ID – Projects database identifier\n"
    "  NOTION_PROJECTS_DATA_SOURCE_ID\n"
    "  NOTION_INBOX_PAGE_ID\n\n"
    "Patch them like this:\n"
    "  @pytest.fixture\n"
    "  def env_vars():\n"
    "      with patch.dict(os.environ, {\n"
    "          'NOTION_TOKEN': 'test-token',\n"
    "          'NOTION_TASKS_DATABASE_ID': 'test-db-id',\n"
    "      }):\n"
    "          yield\n"
    "</notion_env_keys>"
)

JUDGE_SYSTEM = """\
You are an unforgiving, expert-level code evaluation judge. Your standards are extremely high.
You evaluate artifacts produced by an AI coding agent against precise criteria.

SCORING RULES:
- Score each criterion as 0 (complete failure) or 1 (clearly met). There is no partial credit.
- A criterion is met ONLY if you can point to concrete evidence in the artifact. "Probably fine" = 0.
- If the artifact is missing or empty, every criterion scores 0.
- Be adversarial: actively look for ways the artifact FAILS each criterion before deciding it passes.
- Your critique must be brutally specific — quote the exact line or structure that fails.

OUTPUT FORMAT (strict JSON, no extra keys):
{
    "<criterion_key>": {
        "score": 0 or 1,
        "reason": "One sentence: why it passed or the exact failure."
    },
    ...
}

Evaluate EVERY criterion listed. Do not skip any. Do not add criteria not listed.\
"""

RAG_EVALUATOR_SYSTEM = """\
You are a RAG retrieval evaluator for a Notion API assistant.
You will receive a user query and a set of retrieved document chunks.
Your job is to assess how well the retrieved chunks serve the query.

Grade on EXACTLY these four dimensions:

1. topic_matched (bool)
   true  → the majority of retrieved chunks are topically relevant to the query.
   false → the results are off-topic or unrelated.

2. objects_coverage (float, 0.0–1.0)
   Identify every concrete "main object" the user mentions in the query
   (e.g. database, page, block, select property, due date, task).
   Report the fraction of those objects that are meaningfully discussed in the
   retrieved chunks. 1.0 = all objects covered, 0.0 = none covered.

3. endpoint_presence ("present" | "not_present" | "not_needed")
   "present"     → at least one relevant Notion API endpoint (path + method) appears in results.
   "not_present" → the query requires an endpoint but none is returned.
   "not_needed"  → the query is conceptual / definitional and no endpoint is expected.

4. properties_discussed (int N) and properties_total (int K)
   K = total number of distinct Notion properties relevant to answering the query
       that you can identify from the query intent and your knowledge
       (e.g. "Name", "Status", "Due", "Select", "Tags").  Use 0 if no properties apply.
   N = how many of those K properties are actually discussed in the retrieved chunks.
   Report N/K as separate integers; use 0/0 when K=0.

5. critique (str)
   Write exactly 2 sentences:
   • Sentence 1: what the retrieval got right.
   • Sentence 2: the most important gap or weakness.

Respond ONLY with valid JSON matching this exact schema (no extra fields or prose):
{
  "topic_matched":        <bool>,
  "objects_coverage":     <float>,
  "endpoint_presence":    <"present"|"not_present"|"not_needed">,
  "properties_discussed": <int>,
  "properties_total":     <int>,
  "critique":             <string — exactly 2 sentences>
}
"""

RAG_EVAL_PROMPT_BINARY = """\
You are an expert evaluator assessing a RAG-generated [Solution]. Evaluate it using the provided [Query], [Context], and [Real Answer].

Assess the solution across three dimensions using these strict binary rubrics:

1. Query Relevance (0 or 1):
- 1: The solution directly and fully addresses the specific task and properties requested in the query.
- 0: The solution misses the core intent or ignores requested constraints.

2. Context Adherence (0 or 1):
- 1: The solution strictly relies on the provided context. No hallucinated API endpoints, methods, or undocumented parameters are used.
- 0: The solution introduces fabricated information or logic not found in the context.

3. Solution Accuracy (0 or 1):
- 1: The solution's implementation matches the logic, schema, and API structure of the [Real Answer]. Minor variable name differences are acceptable, but the core payload/approach must be functionally identical.
- 0: The solution structurally deviates from the real answer, uses the wrong API schema, or contains functional errors.

Output MUST be valid JSON with this exact schema:
{{
  "reasoning": "step-by-step comparison across query, context, and real answer",
  "scores": {{
    "query_relevance": 0 or 1,
    "context_adherence": 0 or 1,
    "accuracy": 0 or 1
  }}
}}

[Query]
{query}

[Context]
{context}

[Real Answer]
{real_answer}

[Solution]
{solution}
"""

RAG_EVAL_PROMPT_RUBRIC_FULL = """
<system_instruction>
You are a highly critical AI Judge specializing in RAG (Retrieval-Augmented Generation) for the Notion API. Your task is to evaluate the quality of a retrieved context and the resulting code solution against a Query and a Ground-Truth Real Answer.
You must follow the rubrics below strictly. Before providing scores, you must perform a step-by-step analysis of the Signal-to-Noise ratio, Information Sufficiency, and Hallucination presence.
</system_instruction>

<evaluation_rubrics>

<metric id="Q-SN" name="Query Signal-to-Noise">
Goal: Measure how relevant the retrieved context is to the user's query.
- 5: Perfectly Clean. Every sentence in the context directly supports answering the query.
- 4: Mostly Signal. Minor irrelevant info, but the core context is highly focused.
- 3: Mixed. About half the context is relevant; the rest is filler or unrelated Notion docs.
- 2: High Noise. Only a small fraction of the context relates to the query.
- 1: Pure Noise. The context has nothing to do with the query.
</metric>

<metric id="A-SN" name="Answer Signal-to-Noise">
Goal: Measure how much of the [Real Answer]'s technical ingredients exist in the [Context].
- 5: 1:1 Mapping. Every ID, endpoint, and property name in the Real Answer is present in the Context.
- 4: Strong Mapping. Most technical details are present; only generic Python logic is missing.
- 3: Partial. Some IDs or endpoints are present, but others must be inferred or are missing.
- 2: Weak. The context mentions the topic but lacks the specific technical schema used in the Real Answer.
- 1: Zero Mapping. The Context contains none of the technical requirements found in the Real Answer.
</metric>

<metric id="Expressiveness" name="Information Sufficiency">
Goal: Is the [Context] "expressive" enough to reconstruct the [Real Answer] without external help?
- 5: Total Sufficiency. An agent could write the code perfectly using ONLY this context.
- 4: Near Total. Missing only trivial syntax details but contains all Notion-specific logic.
- 3: Sufficient with Assumptions. Requires the agent to use general Notion API knowledge to bridge gaps.
- 2: Insufficient. Context lacks critical IDs or specific property schemas needed for the task.
- 1: Void. No actionable information provided to solve the query.
</metric>

<metric id="Hallucination" name="Strictness/Faithfulness">
Goal: Does the [Solution] invent information or ignore the [Context]/[Real Answer]?
- 5: Perfect Faithfulness. The solution uses exactly what is provided or correctly refuses if info is missing.
- 4: Minor Deviation. Correct logic but used a different variable name or generic placeholder.
- 3: Soft Hallucination. Logic is generally correct, but invented a property name or ID not in Context.
- 2: Hard Hallucination. Invented API endpoints, methods, or logic that contradicts Notion's API.
- 1: Total Fabrication. Ignored the query constraints and the context entirely; code is non-functional.
</metric>

</evaluation_rubrics>

<json_schema>
{{
  "reasoning": "A detailed step-by-step breakdown of signal vs noise, sufficiency of data, and any detected hallucinations.",
  "scores": {{
    "query_signal_to_noise": 1-5,
    "answer_signal_to_noise": 1-5,
    "expressiveness": 1-5,
    "hallucination_strictness": 1-5
  }},
  "verdict": "Pass/Fail (Pass only if all scores >= 4)"
}}
</json_schema>

<task_data>
<query>
{query}
</query>

<context>
{context}
</context>

<real_answer>
{real_answer}
</real_answer>

<solution>
{solution}
</solution>
</task_data>

Respond ONLY with a valid JSON object following the schema provided above.
"""


# ── Prompt Builder Functions ───────────────────────────────────────────────────

def build_multi_query_prompt(query: str, n: int) -> str:
    return (
        f"Original query: {query}\n\n"
        f"Generate {n} alternative formulations of this query. "
        "Each variant must:\n"
        "  • Express the same intent and information need.\n"
        "  • Use different words, phrasing, or emphasis.\n"
        "  • Be a standalone search query (no references to 'the above').\n"
        "  • Stay concise (1–2 sentences max).\n\n"
        'Return ONLY valid JSON: {"queries": ["...", "...", "..."]}'
    )


def build_cot_decompose_prompt(query: str) -> str:
    return (
        f"Query: {query}\n\n"
        "Think step-by-step:\n"
        "  1. What foundational concepts must be understood first?\n"
        "  2. What intermediate facts or steps are required?\n"
        "  3. What is the final specific question?\n\n"
        "Break the query into an ORDERED list of simpler sub-questions. "
        "Each sub-question should be self-contained and searchable. "
        "Earlier items should answer prerequisites for later ones.\n\n"
        'Return ONLY valid JSON: {"sub_questions": ["...", "...", "..."]}'
    )


def build_domain_decompose_prompt(query: str) -> str:
    return (
        f"Query: {query}\n\n"
        "You are an expert on the Notion API (pages, databases, properties, "
        "blocks, data sources, parent objects).\n\n"
        "Identify every Notion-specific concept involved in answering this query "
        "(e.g., 'database parent schema', 'select property structure', "
        "'date property format', 'POST /v1/pages request body').\n\n"
        "For each concept, generate a precise search query that would surface the "
        "relevant documentation or schema details.\n\n"
        'Return ONLY valid JSON: {"queries": ["...", "...", "..."]}'
    )


def build_analyze_requirements_prompt(user_prompt: str) -> str:
    return (
        f"User Request: {user_prompt}\n\n"
        "Analyze this request and identify the TOP 5 most critical things that must be "
        "extracted from the Notion API documentation to implement this correctly.\n\n"
        "For each item, provide:\n"
        "  • The name of the concept/endpoint/property (key)\n"
        "  • A one-sentence description of why it's essential (value)\n\n"
        "Focus on:\n"
        "  - API endpoints required\n"
        "  - Data source / database specifics\n"
        "  - Field/property names and types\n"
        "  - Request/response schemas\n"
        "  - Authentication or parameter patterns\n\n"
        'Return ONLY valid JSON: {"requirement_name": "why it matters", ...}'
    )


def build_generate_request_plan_prompt(user_prompt: str, rag_context: str) -> str:
    return (
        f"<user_request>{user_prompt}</user_request>\n\n"
        f"<api_description>{rag_context}</api_description>\n\n"
        "Write a concise bullet-point plan (≤10 bullets) of every concrete step required "
        "to implement the user request as a Python function using the Notion API. "
        "Focus on what parameters to accept, which endpoint(s) to call, how to build "
        "the request body, and what to return. Be specific — no filler text."
    )


def build_generate_tests_draft_prompt(general_info: str) -> str:
    return (
        f"{general_info}\n\n"
        f"{GENERATE_TESTS_KEYS_CONTEXT}\n\n"
        "Write Python test cases using pytest for the function described in <user_request>.\n"
        "HARD RULES — any violation makes the test file unusable:\n"
        "  1. Import the target function from 'solution' (e.g. `from solution import add_task`).\n"
        "  2. Mock ALL HTTP calls with unittest.mock.patch (requests.post / requests.get / etc.).\n"
        "  3. Patch env vars with unittest.mock.patch.dict(os.environ, {...}) — NEVER pass raw secrets.\n"
        "  4. Do NOT instantiate NotionKeys or pass notion_keys objects; the function reads os.getenv internally.\n"
        "  5. Cover: happy path, missing required field (should raise), API 4xx/5xx error response.\n"
        "  6. Use EXACT field names and endpoint URLs from <api_context> in assertions.\n"
        "  7. Output only valid Python — no markdown fences.\n\n"
        'Return as JSON: {"test_code": "<full python code>"}'
    )


def build_generate_tests_grade_prompt(general_info: str, candidates_block: str) -> str:
    return (
        f"{general_info}\n\n"
        "You are a senior Python test engineer reviewing three test drafts.\n\n"
        f"{candidates_block}\n\n"
        "Select or merge the best tests. REJECT any code that:\n"
        "  • Hardcodes a real token, database ID or any Notion secret.\n"
        "  • Passes a NotionKeys object or dict of secrets directly to the function.\n"
        "  • Does not patch os.environ for every env-var the function needs.\n"
        "  • Has import errors, undefined names, or duplicate test logic.\n\n"
        "REQUIREMENTS for the final file:\n"
        "  • All imports at the top (os, pytest, unittest.mock, etc.).\n"
        "  • All HTTP calls mocked with unittest.mock.patch.\n"
        "  • All env vars mocked with unittest.mock.patch.dict(os.environ, {...}).\n"
        "  • Runnable with `pytest current_tests.py` without modification.\n\n"
        'Return as JSON: {"test_code": "<complete runnable code>", "reasoning": "<brief>"}'
    )


def build_generate_code_prompt(
    general_info: str,
    test_code: str,
    feedback: Optional[str] = None,
) -> str:
    feedback_block = (
        f"\n\n<judge_feedback>\n{feedback}\n</judge_feedback>\n"
        "Use <judge_feedback> as the primary repair spec.\n"
        "First identify the root cause from feedback, then implement the HOW_TO_FIX steps exactly.\n"
        "Do not re-argue or re-evaluate feedback; directly apply it in code.\n"
        if feedback else ""
    )
    return (
        f"{general_info}\n"
        f"{feedback_block}\n"
        f"{CODEGEN_ENV_CONTEXT}\n\n"
        "<tests_to_pass>\n"
        f"{test_code}\n"
        "</tests_to_pass>\n\n"
        "Write a complete Python module (saved as solution.py) that:\n"
        "  • Implements the function(s) described in <user_request>.\n"
        "  • Uses ONLY the endpoints and field names in <api_context> — no invented URLs or fields.\n"
        "  • Passes every test in <tests_to_pass>.\n\n"
        "SECURITY — MANDATORY, no exceptions:\n"
        "  • First two imports MUST be: `import os` then `import dotenv; dotenv.load_dotenv()`.\n"
        "  • ALL secrets (tokens, IDs) MUST come from os.getenv('ENV_VAR_NAME').\n"
        "  • Valid patterns:\n"
        "      def my_fn(token: str = os.getenv('NOTION_TOKEN'), db_id: str = os.getenv('NOTION_TASKS_DATABASE_ID'))\n"
        "      token = os.getenv('NOTION_TOKEN')  # inside function body\n"
        "  • FORBIDDEN: ANY hardcoded string that looks like a token, UUID, or database ID.\n"
        "  • FORBIDDEN: Accepting a NotionKeys dataclass or dict of secrets as a parameter.\n\n"
        "CODE QUALITY:\n"
        "  • Docstrings and type hints on every public function.\n"
        "  • Use the `requests` library for HTTP calls.\n"
        "  • Every HTTP request header dict MUST include `'Notion-Version': '2022-06-28'`.\n"
        "  • Raise a descriptive Exception on any non-2xx API response, but print the 'e.response.text' for debugging.\n"
        "  • `if __name__ == '__main__':` block that calls the function with os.getenv values.\n"
        "  • The `if __name__ == '__main__':` block MUST call `sys.exit(1)` if the function raises any exception or returns an error state.\n"
        "  • Never leave default mutable objects as function arguments (e.g., `def fn(data: list = [])`); use `None` instead.\n\n"
        'Return as JSON: {"code": "<complete python module>", "function_name": "<main function name>"}'
    )


def build_preflect_prompt(
    general_info: str,
    generated_code: str,
    test_summary: str,
    sol_summary: str,
) -> str:
    return (
        f"{general_info}\n\n"
        "<generated_code>\n"
        f"{generated_code}\n"
        "</generated_code>\n\n"
        "<test_results>\n"
        f"{test_summary}\n"
        "</test_results>\n\n"
        f"{sol_summary}"
        "You are about to diagnose a code failure against the Notion API.\n"
        "Before doing so: decide if the <api_context> already contains enough information "
        "to pinpoint the exact failing endpoint, field names, or schema — or if you need "
        "to look up additional Notion API documentation.\n\n"
        "If you need more information, specify up to 3 precise RAG queries that would return "
        "the missing Notion API details (e.g., 'POST /v1/pages request body properties', "
        "'tasks database select property options', 'Notion block children append endpoint').\n"
        "Only request a lookup if genuinely needed — do not look up things already answered "
        "by <api_context>.\n\n"
        'Return JSON: {"needs_lookup": bool, "queries": ["...", ...]}'
    )


def build_reflect_code_prompt(
    general_info: str,
    extra_ctx_block: str,
    generated_code: str,
    test_summary: str,
    sol_summary: str,
) -> str:
    return (
        f"{general_info}\n\n"
        f"{extra_ctx_block}"
        "<generated_code>\n"
        f"{generated_code}\n"
        "</generated_code>\n\n"
        "<test_results>\n"
        f"{test_summary}\n"
        "</test_results>\n\n"
        f"{sol_summary}"
        "You are a failure-analysis assistant for iterative code repair.\n"
        "DO NOT grade quality or assign scores. Focus only on diagnostic feedback for fixing the next attempt.\n\n"
        "ANALYSIS SEQUENCE:\n"
        "1) ERROR_DIAGNOSIS: Closely read ALL of <test_results> and <solution_run> (stdout, stderr, exit codes).\n"
        "   Pinpoint the exact error messages, assertion failures, or runtime exceptions.\n"
        "2) API_MISCONCEPTIONS: Check against <api_context> and <reflection_context> (if present). Common issues:\n"
        "   - Wrong endpoint URL or HTTP method\n"
        "   - Incorrect request field names or payload structure\n"
        "   - Missing required headers (e.g., Authorization, Content-Type)\n"
        "   - Loading secrets incorrectly or not at all\n"
        "   - Misunderstanding Notion's nested object schemas\n"
        "3) RAG_CONTEXT_VERIFY: Re-scan <api_context> and <reflection_context> for exact field names, endpoint, "
        "and required properties. If code diverges from either, that's the root cause.\n\n"
        "Set 'pass' to true only if ALL of these hold:\n"
        "  1. Tests passed or undefined (exit_code == 0).\n"
        "  2. Code correctly implements the user request and matches the Notion API schema in <api_context>.\n"
        "  3. Code is runnable (no syntax/runtime errors) and reasonably modular.\n\n"
        "If 'pass' is false, feedback must contain exactly these sections:\n"
        "1) WHAT_IS_GOING_WRONG: Concrete failing behaviors tied to tests or API mismatches.\n"
        "2) ROOT_CAUSE: Is it an API misconception (wrong endpoint/field/header) or a logic error?\n"
        "3) HOW_TO_FIX: Ordered, code-level steps. Reference exact field names and schemas from RAG.\n"
        "4) DONE_WHEN: Objective checks confirming the fix is correct.\n\n"
        "Keep feedback specific, terse, and implementation-oriented.\n"
        'Return as JSON: {"reasoning": "1-2 sentences summarising what the evidence shows", "pass": bool, "feedback": "..."}'
    )


def build_judge_category_prompt(
    category: str,
    criteria: Dict[str, str],
    user_query: str,
    artifact_text: str,
    rag_data_str: str = "",
) -> str:
    criteria_block = "\n".join(f"  - {k}: {v}" for k, v in criteria.items())
    rag_section = f"## RAG Context (reference)\n{rag_data_str}\n\n" if rag_data_str else ""
    return (
        f"{JUDGE_SYSTEM}\n\n"
        f"## Category: {category.upper()}\n\n"
        f"## User Query\n{user_query}\n\n"
        f"{rag_section}"
        f"## Artifact to Evaluate\n```\n{artifact_text}\n```\n\n"
        f"## Criteria\n{criteria_block}\n\n"
        "Evaluate now. Return ONLY the JSON object."
    )


def build_concision_prompt(max_words: int) -> str:
    """
    Build a self-enforcing word constraint that the model applies to itself.
    
    This instruction is appended to the user's message and frames a hard constraint
    on the MODEL'S response generation. The model must:
      1. Read this as a binding directive (not content to relay back)
      2. Apply the word limit to THIS RESPONSE's actual output
      3. Never echo or repeat this constraint in the response
    
    Args:
        max_words: The target maximum word count for the response.
        
    Returns:
        Imperative constraint directive for the model's own behavior.
    """
    return (
        f"[CONSTRAINT FOR THIS RESPONSE]\n"
        f"YOU MUST limit your response to UNDER {max_words} words.\n"
        f"This constraint applies to your output—do not mention or repeat it.\n"
        f"Prioritize core content; trim examples, extra detail, and preamble.\n"
        f"Do not acknowledge this instruction; proceed directly with the task."
    )


# ── Evaluation Criteria ─────────────────────────────────────────────────────────────────
# All categories use the same adversarial LLM judge.
# "rag" is evaluated against the retrieved context text, like any other artifact.

EVAL_CRITERIA = {
    "rag": {
        "model": "gemma12",
        "criteria": {
            "topic_matched":      "The majority of retrieved chunks are clearly on-topic for the user query — not generic, off-topic, or loosely related sections.",
            "objects_coverage":   "The key Notion objects explicitly mentioned in the query (e.g. page, database, block, property type) are substantively discussed in the retrieved chunks.",
            "endpoint_presence":  "At least one relevant Notion API endpoint URL or HTTP method appears in the retrieved chunks, or the query is purely conceptual and no endpoint is needed.",
            "properties_covered": "The specific property names and payload fields needed to fulfil the query are mentioned in the retrieved chunks — not just the object type in general.",
        },
    },
    "plan": {
        "model": "gemma27",
        "criteria": {
            "plan_is_direct":     "The plan directly addresses the user query without unnecessary preamble or tangents.",
            "plan_is_actionable": "Every bullet point describes a concrete, implementable step — not vague advice.",
            "plan_is_concise":    "The plan is tight — no redundant bullets, no filler text, no repeated information.",
            "plan_matches_query": "The plan fully covers the user's actual request — no missing steps, no hallucinated extras.",
        },
    },
    "tests": {
        "model": "gemma12",
        "criteria": {
            "tests_relevant_to_problem":      "Each test targets behaviour described in the user query — no irrelevant or generic tests.",
            "tests_match_rag_data":           "Tests use the exact endpoint URLs, field names and schemas from the RAG context — not invented ones.",
            "tests_cover_edge_cases":         "Tests include at least: missing required fields, API error responses, and boundary values.",
            "tests_cover_core_functionality": "The happy-path test validates the complete request body structure, headers and return value.",
            "tests_file_is_runnable":         "The test file is syntactically valid Python 3, all imports resolve, and pytest can collect every test.",
            "tests_pass_when_executed":       "When tests are enabled, pytest exits successfully (exit_code == 0).",
        },
    },
    "code": {
        "model": "gemma27",
        "criteria": {
            "code_matches_query":    "The function implements exactly what the user asked for — no more, no less.",
            "code_matches_rag_data": "Endpoint URLs, HTTP methods, request body structure and field names match the RAG context precisely.",
            "code_is_concise":       "No dead code, no commented-out blocks, no unnecessary abstractions for a single-purpose function.",
            "code_is_modular":       "Logic is cleanly separated: header construction, body construction, HTTP call, error handling are distinct.",
            "code_doesnt_leak_data": "No hardcoded API tokens, database IDs or secrets anywhere in the source — zero tolerance.",
            "code_uses_os_getenv":   "All sensitive keys are loaded via os.getenv() — either at module level or as function parameters defaulting to os.getenv().",
            "code_is_runnable":      "The file is syntactically valid, imports succeed, and the function can be called without crashing on valid inputs.",
            "code_is_documented":    "Public functions include meaningful docstrings and type hints.",
        },
    },
    "reflection": {
        "model": "gemma12",
        "criteria": {
            "correctly_understands_traces":       "The reflection accurately identifies what went wrong (or right) based on test output and error traces.",
            "actionable_feedback":                "Every piece of feedback suggests a specific code change — not vague observations like 'improve error handling'.",
            "feedback_helps_solve_errors":        "Following the feedback would actually fix the identified issues — not introduce new ones.",
            "feedback_identifies_critical_error": "The most critical/blocking error is called out first; minor style issues are deprioritised or omitted.",
            "checks_user_request_correctness":    "Feedback verifies whether the implementation truly matches the user request.",
            "checks_api_schema_correctness":      "Feedback verifies API endpoint/body/schema correctness against the provided Notion context.",
            "checks_runtime_safety":              "Feedback explicitly flags syntax/runtime blockers that prevent successful execution.",
            "checks_code_modularity":             "Feedback highlights missing separation of concerns when modularity is inadequate.",
        },
    },
}


def build_prompt_statements(context: str, statements: List[str]) -> str:
    output_format = """
    ### Output Format
    `[
    {
        "statement": "The original statement text",
        "status": "Present | Wrong | Not present",
        "evidence": "Verbatim quote from the context",
        "reasoning": "Brief explanation of the status"
    }
    ]`
    """
    
    
    return """
    You are a **Scrupulous Technical Auditor**. Your goal is to evaluate a provided context for the technical completeness of specific API requirements using a "Quote-before-Judge" methodology.

    ### Inputs

    - **Context to Inspect:** {context}
    - **Technical Statements:** {statements}

    ### Evaluation Rubric

    For each statement, you must determine one of the following statuses:

    1. **Present:** The context contains the exact technical value, structure, or endpoint required.
    2. **Wrong:** An incorrect value, endpoint, or HTTP method is used (e.g., POST instead of PATCH). **Note:** Ignore stylistic differences like variable names; focus only on literal API syntax and values.
    3. **Not present:** The context fails to mention this technical detail entirely.

    ### Instructions

    1. **Verbatim Extraction:** For every statement, attempt to extract the literal code snippet or string from the {{context}} that relates to the requirement.
    2. **Strict Audit:** If the exact token or structure is not found, it must be marked Not present. If a conflicting token is found, it must be marked Wrong.
    3. **JSON Output:** Return a JSON list of objects.

    ### Constraints

    - **Grounding:** You must provide the evidence (the verbatim quote) for every "Present" or "Wrong" status. If the status is "Not present", the evidence must be "NONE".
    - **Brevity:** Keep the reasoning field under 15 words.
    - **Accuracy:** Pierce the "blurred cloud" of general descriptions; look for literal strings (URLs, JSON keys, headers).
    
    {output_format}
    """.format(context=context, statements="\n- ".join(statements), output_format=output_format)
    