# ── Utils ──────────────────────────────────────────────────────────────────────────────
import asyncio
import glob
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from functools import partial
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openai
import yaml
from langsmith.wrappers import wrap_openai

from .config import (
    _MODEL_MAP,
)
from .prompts import (
    EVAL_CRITERIA,
    build_analyze_requirements_prompt,
    build_generate_code_prompt,
    build_generate_request_plan_prompt,
    build_generate_tests_draft_prompt,
    build_generate_tests_grade_prompt,
    build_judge_category_prompt,
    build_preflect_prompt,
    build_reflect_code_prompt,
)
from .rag_utils import (
    query_qdrant,
)

logger = logging.getLogger(__name__)


def load_eval_tasks(evals_dir: str = "evals") -> Dict[str, Dict[str, Any]]:
    """Load evaluation tasks from YAML files. Returns {} if directory not found."""
    tasks: Dict[str, Dict[str, Any]] = {}
    for yaml_path in sorted(glob.glob(os.path.join(evals_dir, "*.yaml"))):
        stem = Path(yaml_path).stem
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        tasks[stem] = data if data else {}
    return tasks

def extract_json_from_response(response_content: Optional[str]) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM response content.
    
    Handles multiple formats:
    1. Direct JSON string
    2. JSON wrapped in markdown code blocks (```json ... ```)
    3. JSON with extra whitespace/newlines
    4. JSON embedded in text (finds leftmost { or [ and rightmost } or ])
    
    Args:
        response_content: Raw response content from LLM
        
    Returns:
        Parsed JSON as dictionary or list
        
    Raises:
        ValueError: If JSON cannot be extracted or parsed
    """
    if response_content is None:
        raise ValueError("Response content is None")
    
    # Try direct parsing first
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        logger.debug("Direct JSON parsing failed, attempting markdown extraction")
    
    # Try extracting from markdown code blocks
    # Pattern matches: ```json\n{...}\n``` or ```\n{...}\n```
    patterns = [
        r'```json\s*\n(.*?)\n```',  # ```json ... ```
        r'```\s*\n(.*?)\n```',       # ``` ... ```
        r'```json\s*(.*?)```',       # ```json...``` (no newlines)
        r'```\s*(.*?)```',           # ```...``` (no newlines)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse JSON from pattern: {pattern}")
                continue
    
    # Fallback: Try the split method from notebook
    # result.split('```')[1][5:] - splits by ``` and takes second part, skipping "json\n"
    try:
        parts = response_content.split('```')
        if len(parts) >= 3:  # Should have at least [before, content, after]
            json_str = parts[1]
            # Remove "json" prefix if present
            if json_str.startswith('json'):
                json_str = json_str[4:].strip()
            return json.loads(json_str)
    except (json.JSONDecodeError, IndexError) as e:
        logger.debug(f"Fallback split method failed: {e}")
    
    # Advanced fallback: Find leftmost { or [ and rightmost } or ]
    try:
        left_brace = response_content.find('{')
        left_bracket = response_content.find('[')
        right_brace = response_content.rfind('}')
        right_bracket = response_content.rfind(']')
        
        # Determine which bracket pair to use
        start_idx = -1
        end_idx = -1
        
        if left_brace >= 0 and right_brace > left_brace:
            if left_bracket >= 0 and left_bracket < left_brace:
                # [ comes before {, use [ ... ]
                if right_bracket > right_brace:
                    start_idx = left_bracket
                    end_idx = right_bracket
                else:
                    start_idx = left_brace
                    end_idx = right_brace
            else:
                # { comes first, use { ... }
                start_idx = left_brace
                end_idx = right_brace
        elif left_bracket >= 0 and right_bracket > left_bracket:
            # Only [ ... ] is valid
            start_idx = left_bracket
            end_idx = right_bracket
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_content[start_idx:end_idx + 1]
            try:
                result = json.loads(json_str)
                logger.debug("Successfully extracted JSON using bracket-matching method")
                return result
            except json.JSONDecodeError as e:
                logger.debug(f"Bracket-matching extraction failed: {e}")
    except Exception as e:
        logger.debug(f"Bracket-matching method error: {e}")
    
    # If all methods fail, raise error with helpful message
    raise ValueError(
        f"Failed to extract JSON from response. "
        f"Content preview: {response_content[:200]}..."
    )





# ── Pipeline ─────────────────────────────────────────────────────────────────────────────

_async_client = wrap_openai(openai.AsyncOpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    max_retries=25,
))

def _check_finish_reason(model_name: str, finish_reason: str) -> None:
    """Check and log the finish_reason from LLM response."""
    print(f"[async_chat_wrapper] Model: {model_name}, finish_reason: {finish_reason}")
    if finish_reason != "stop":
        logger.warning(f"Non-stop finish_reason: {finish_reason} (response may be truncated)")

async def async_chat_wrapper(
    messages: list,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    json_output: bool = False,
    model_size: str = "gemma27",
) -> Any:
    model_name = _MODEL_MAP.get(model_size)
    if model_name is None:
        model_name = model_size  
        logger.warning(f"Model size '{model_size}' not found in map, using as-is: {model_name}")
        
    msgs = list(messages)
    if json_output:
        if 'gemini' in model_name:
            response = await _async_client.chat.completions.parse(
                model=model_name,
                messages=msgs,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            _check_finish_reason(model_name, response.choices[0].finish_reason)
            content = response.choices[0].message.content
            logger.debug(f"LLM response: {content}")
            return json.loads(content)
        
        else:
            msgs.append({"role": "user", "content": "Please provide the output in JSON format."})
         
    response = await _async_client.chat.completions.create(
        model=model_name,
        messages=msgs,
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    _check_finish_reason(model_name, response.choices[0].finish_reason)
    
    content = response.choices[0].message.content
    logger.debug(f"LLM response: {content}")
    if json_output:
        content = extract_json_from_response(content)
    return content
# ── Step 1.5: Requirements Analysis ────────────────────────────────────────────────
async def analyze_requirements(user_prompt: str) -> Dict[str, str]:
    """
    Pre-RAG analysis using a medium model to extract key requirements.
    
    This lightweight step identifies the top documentation/API concepts needed
    before dumping the full corpora, making RAG retrieval more targeted.
    
    Returns a dict with requirement names as keys and descriptions as values.
    """
    prompt = build_analyze_requirements_prompt(user_prompt=user_prompt)
    
    result: Dict[str, str] = await async_chat_wrapper(
        [{"role": "user", "content": prompt}],
        json_output=True, max_tokens=600, temperature=0.2, model_size="gemma12",
    )
    
    logger.info("Requirements analysis: %s", result)
    return result


# ── Step 3: General Info ────────────────────────────────────────────────────────────────
def build_general_info(user_prompt: str, rag_context: str, request_plan: str) -> str:
    """
    Assemble a structured context block reused across steps 3-5.

    Args:
        user_prompt:  Original user task.
        rag_context:  Plain-text RAG context (summary or raw chunks concatenated).
        request_plan: Bullet-point implementation plan from generate_request_plan.

    Uses XML tags and places most-critical data at the start/end (lost-in-middle mitigation).
    """
    return (
        # Most-important anchor at the top
        "<user_request>\n"
        f"{user_prompt}\n"
        "</user_request>\n\n"

        "<request_plan>\n"
        f"{request_plan}\n"
        "</request_plan>\n\n"

        # RAG context in the middle
        "<api_context>\n"
        f"{rag_context}\n"
        "</api_context>\n\n"

        # Repeat key request at the end for recency bias
        "<reminder>Implement exactly what is described in <user_request>. "
        "Use only the endpoints and schemas provided in <api_context>.</reminder>\n"
    )


async def generate_request_plan(user_prompt: str, rag_context: str) -> str:
    """Bullet-point plan of what needs to be implemented — large model."""
    prompt = build_generate_request_plan_prompt(user_prompt=user_prompt, rag_context=rag_context)
    return await async_chat_wrapper(
        [{"role": "user", "content": prompt}],
        json_output=False, max_tokens=500, temperature=0.3, model_size="gemma27",
    )


# ── Step 4: Generate Tests ────────────────────────────────────────────────────────────────
_TEST_SCHEMA = {
    "type": "object",
    "properties": {
        "test_code": {"type": "string", "description": "Complete, runnable pytest code"}
    },
    "required": ["test_code"]
}

_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "test_code": {"type": "string"},
        "reasoning": {"type": "string"}
    },
    "required": ["test_code", "reasoning"]
}


async def generate_tests(general_info: str) -> str:
    """
    TDD step — 3 parallel different-model drafts, then 1 medium-model grader
    picks / merges the best result.

    Returns ready-to-write Python test code (string).
    """
    draft_prompt = build_generate_tests_draft_prompt(general_info=general_info)

    # 3 parallel different-size drafts
    drafts: list[Dict] = await asyncio.gather(*[
        async_chat_wrapper(
            [{"role": "user", "content": draft_prompt}],
            json_output=True, max_tokens=2200, temperature=0.7, model_size="gemma1",
        ),
        async_chat_wrapper(
            [{"role": "user", "content": draft_prompt}],
            json_output=True, max_tokens=2200, temperature=0.7, model_size="gemma4",
        ),
        async_chat_wrapper(
            [{"role": "user", "content": draft_prompt}],
            json_output=True, max_tokens=2200, temperature=0.7, model_size="gemma12",
        ),
    ])

    candidates_block = "\n\n".join(
        f"<candidate_{i+1}>\n{d.get('test_code', '')}\n</candidate_{i+1}>"
        for i, d in enumerate(drafts)
    )

    grade_prompt = build_generate_tests_grade_prompt(
        general_info=general_info,
        candidates_block=candidates_block,
    )

    graded: Dict = await async_chat_wrapper(
        [{"role": "user", "content": grade_prompt}],
        json_output=True, max_tokens=2600, temperature=0.2, model_size="gemma27",
    )

    logger.info("Test grader reasoning: %s", graded.get("reasoning", ""))
    return graded.get("test_code", "")


def write_tests(test_code: str, path: str = "current_tests.py") -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(test_code)
    print(f"Tests written → {path}")


# ── Step 5: Generate Code ────────────────────────────────────────────────────────────────
_CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Complete Python solution code"},
        "function_name": {"type": "string"}
    },
    "required": ["code", "function_name"]
}


async def generate_code(
    general_info: str,
    test_code: str,
    feedback: Optional[str] = None,
    model_size: str = "gemma27",
    temperature: float = 0.3,
    max_tokens: int = 2500,
) -> Dict[str, str]:
    """
    Generate the solution code.

    Args:
        general_info: Assembled context from step 3.
        test_code:    The tests the code must pass (from step 4).
        feedback:     Optional judge feedback from a previous failed attempt.
        model_size:   Model alias key from _MODEL_MAP.
        temperature:  Sampling temperature.
        max_tokens:   Maximum tokens for generation.

    Returns:
        {"code": "...", "function_name": "..."}
    """
    prompt = build_generate_code_prompt(
        general_info=general_info,
        test_code=test_code,
        feedback=feedback,
    )

    result: Dict = await async_chat_wrapper(
        [{"role": "user", "content": prompt}],
        json_output=True, max_tokens=max_tokens, temperature=temperature, model_size=model_size,
    )
    return result


def write_solution(code: str, path: str = "solution.py") -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(code)
    print(f"Solution written → {path}")


# ── Step 6: Judge + Reflect ─────────────────────────────────────────────────────────────

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "1-2 sentence summary of what the evidence shows before reaching a verdict"
        },
        "pass": {"type": "boolean"},
        "feedback": {
            "type": "string",
            "description": "Failure analysis and concrete fix guidance for next code attempt"
        }
    },
    "required": ["reasoning", "pass", "feedback"]
}

_PREFLECT_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_lookup": {"type": "boolean"},
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
            "description": "RAG queries to fetch more Notion API docs. Empty if needs_lookup=false."
        }
    },
    "required": ["needs_lookup", "queries"]
}


def run_tests(test_path: str = "current_tests.py") -> Dict[str, Any]:
    """Run pytest as a subprocess, return stdout/stderr + exit code."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"],
        capture_output=True, text=True,
    )
    return {
        "exit_code": result.returncode,
        "stdout":    result.stdout,
        "stderr":    result.stderr,
        "passed":    result.returncode == 0,
    }


async def _reflect_search(query: str) -> str:
    """
    Search Qdrant (top_k=1) for a single query and auto-summarize with a small model.
    """
    results = await query_qdrant(
        query=query,
        collection_name="notion_docs_leaf",
        top_k=1,
        threshold=0.3,
    )
    if not results:
        return f"[No results for: {query}]"

    raw_text = results[0].text
    summary = await async_chat_wrapper(
        [{"role": "user", "content": (
            f"Summarize the following Notion API documentation snippet in 3-5 sentences. "
            f"Keep all endpoint URLs, field names, and required properties verbatim.\n\n{raw_text}"
        )}],
        json_output=False, max_tokens=300, temperature=0.1, model_size="gemma4",
    )
    return f"[Query: {query}]\n{summary}"


async def _preflect(
    general_info: str,
    generated_code: str,
    test_summary: str,
    sol_summary: str,
) -> Dict[str, Any]:
    """
    Lightweight first pass: decide if extra RAG lookups are needed and which queries.
    Returns {"needs_lookup": bool, "queries": [...]} — max 3 queries.
    """
    prompt = build_preflect_prompt(
        general_info=general_info,
        generated_code=generated_code,
        test_summary=test_summary,
        sol_summary=sol_summary,
    )
    return await async_chat_wrapper(
        [{"role": "user", "content": prompt}],
        json_output=True, max_tokens=300, temperature=0.1, model_size="gemma12",
    )


async def reflect_code(
    general_info: str,
    generated_code: str,
    test_results: Dict[str, Any],
    solution_run: Optional[Dict[str, Any]] = None,
    reflection_context: Optional[List[str]] = None,
    model_size: str = "gemma27",
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> Dict[str, Any]:
    """
    LLM reflection step that diagnoses failures and gives repair guidance.

    1. Runs a lightweight preflect pass to decide if extra RAG lookups are needed.
     2. If yes, fetches up to 3 targeted RAG snippets (top_k=1, gemma4 summary),
         accumulated in the reflection_context list.
    3. Runs the full reflection with all available context injected.

    Returns {"reasoning": str, "pass": bool, "feedback": "..."}
    """
    test_summary = (
        f"Exit code: {test_results['exit_code']}\n"
        f"Passed: {test_results['passed']}\n\n"
        f"--- stdout ---\n{test_results['stdout'][-3000:]}\n"
        f"--- stderr ---\n{test_results['stderr'][-1000:]}"
    )

    sol_summary = ""
    if solution_run:
        sol_summary = (
            "<solution_run>\n"
            f"Exit code: {solution_run['exit_code']}\n"
            f"--- stdout ---\n{solution_run['stdout'][-2000:]}\n"
            f"--- stderr ---\n{solution_run['stderr'][-1000:]}\n"
            "</solution_run>\n\n"
        )

    '''# ── Step 1: Preflect — decide if extra RAG lookups needed ────────────────
    preflect = await _preflect(general_info, generated_code, test_summary, sol_summary)
    needs_lookup = preflect.get("needs_lookup", False)
    lookup_queries = preflect.get("queries", [])[:3]

    if reflection_context is None:
        reflection_context = []

    if needs_lookup and lookup_queries:
        print(f"  [reflect] RAG lookup needed — {len(lookup_queries)} quer{'y' if len(lookup_queries)==1 else 'ies'}: {lookup_queries}")
        for q in lookup_queries:
            reflection_context.append(await _reflect_search(q))
    else:
        print("  [reflect] No extra RAG lookup needed.")'''

    # ── Step 2: Build extra context block from accumulated reflection context ─
    extra_ctx_block = ""
    if reflection_context:
        joined = "\n\n".join(reflection_context)
        extra_ctx_block = (
            "<reflection_context>\n"
            "Additional Notion API documentation fetched during reflection:\n\n"
            f"{joined}\n"
            "</reflection_context>\n\n"
        )

    # ── Step 3: Full reflection ───────────────────────────────────────────────
    prompt = build_reflect_code_prompt(
        general_info=general_info,
        extra_ctx_block=extra_ctx_block,
        generated_code=generated_code,
        test_summary=test_summary,
        sol_summary=sol_summary,
    )

    return await async_chat_wrapper(
        [{"role": "user", "content": prompt}],
        json_output=True, max_tokens=max_tokens, temperature=temperature, model_size=model_size,
    )


# ── Evaluation ─────────────────────────────────────────────────────────────────────────────


async def _judge_single_category(
    category: str,
    criteria: Dict[str, str],
    model_size: str,
    artifact_text: str,
    user_query: str,
    rag_data_str: str = "",
) -> Dict[str, Any]:
    """LLM judge for one evaluation category. Returns {criterion: {score, reason}}."""
    prompt = build_judge_category_prompt(
        category=category,
        criteria=criteria,
        user_query=user_query,
        artifact_text=artifact_text,
        rag_data_str=rag_data_str,
    )

    return await async_chat_wrapper(
        [{"role": "user", "content": prompt}],
        json_output=True, max_tokens=600, temperature=0.1, model_size=model_size,
    )


async def run_all_evaluations(
    pipeline_artifacts: Dict[str, Any],
    user_query: str,
    eval_criteria: Dict[str, Dict] = EVAL_CRITERIA,
) -> Dict[str, Any]:
    """
    Run ALL evaluation categories on artifacts from a single pipeline run.

    All categories — including "rag" — use the adversarial LLM judge
    (_judge_single_category). The "rag" artifact is the retrieved context text.

    Returns:
    {
        "rag":        {criterion: {"score": 0|1, "reason": …}, …},
        "plan":       {…},
        "tests":      {…},
        "code":       {…},
        "reflection": {…},
        "summary": {
            "rag_score": float, "plan_score": float, "tests_score": float,
            "code_score": float, "reflection_score": float, "overall_score": float
        }
    }
    """
    rag_context  = pipeline_artifacts.get("rag_context", "")
    last_trial   = (pipeline_artifacts.get("trials") or [{}])[-1]

    # Build reflection text from per-trial judge feedback
    reflection_parts = [
        f"Trial {t['trial_num']} feedback:\n{t['verdict'].get('feedback', '')}"
        for t in pipeline_artifacts.get("trials", [])
        if t.get("verdict", {}).get("feedback")
    ]
    reflection_text = "\n\n".join(reflection_parts) or "(no feedback produced)"

    artifact_map = {
        "rag":        rag_context or "(no RAG context produced)",
        "plan":       pipeline_artifacts.get("request_plan", "(no plan produced)"),
        "tests":      pipeline_artifacts.get("test_code",    "(no tests produced)"),
        "code":       last_trial.get("code", pipeline_artifacts.get("final_code", "(no code produced)")),
        "reflection": reflection_text,
    }

    # ── All categories: parallel LLM judges ──────────────────────────────────
    judge_tasks = [
        _judge_single_category(
            category=cat,
            criteria=eval_criteria[cat]["criteria"],
            model_size=eval_criteria[cat]["model"],
            artifact_text=artifact_map.get(cat, ""),
            user_query=user_query,
            rag_data_str=rag_context[:3000] if cat != "rag" else "",
        )
        for cat in eval_criteria
    ]
    judge_results = await asyncio.gather(*judge_tasks)

    eval_results: Dict[str, Any] = {}
    for cat, raw in zip(eval_criteria.keys(), judge_results):
        eval_results[cat] = raw

    # ── Compute per-category scores ───────────────────────────────────────────
    summary: Dict[str, float] = {}
    for category, config in eval_criteria.items():
        criteria_keys = list(config["criteria"].keys())
        scores = [
            eval_results.get(category, {}).get(k, {}).get("score", 0)
            if isinstance(eval_results.get(category, {}).get(k), dict) else 0
            for k in criteria_keys
        ]
        summary[f"{category}_score"] = round(sum(scores) / max(len(criteria_keys), 1), 3)

    summary["overall_score"] = round(
        sum(v for k, v in summary.items() if k.endswith("_score")) / len(eval_criteria), 3
    )
    eval_results["summary"] = summary

    return eval_results


print("Evaluation functions defined.")


def plot_evaluation_results(
    all_results: Dict[str, Dict[str, Any]],
    eval_criteria: Dict[str, Dict] = EVAL_CRITERIA,
) -> None:
    """
    Visualize evaluation results across all eval tasks.
    
    Args:
        all_results: Dict mapping task_id -> eval_results (output of run_all_evaluations).
        eval_criteria: The criteria definitions for axis labels.
    
    Produces:
        1. Grouped bar chart — per-criterion scores across tasks
        2. Radar chart — category-level scores per task
        3. Summary heatmap — tasks x categories
    """
    task_ids = list(all_results.keys())
    categories = [c for c in eval_criteria.keys()]
    
    if not task_ids:
        print("No results to plot.")
        return
    
    # ── Figure 1: Summary Heatmap (tasks × categories) ───────────────────
    fig1, ax1 = plt.subplots(figsize=(8, max(3, len(task_ids) * 1.2)))
    
    heatmap_data = []
    for tid in task_ids:
        row = []
        summary = all_results[tid].get("summary", {})
        for cat in categories:
            row.append(summary.get(f"{cat}_score", 0.0))
        heatmap_data.append(row)
    
    heatmap_arr = np.array(heatmap_data)
    im = ax1.imshow(heatmap_arr, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels([c.title() for c in categories], rotation=45, ha="right")
    ax1.set_yticks(range(len(task_ids)))
    ax1.set_yticklabels(task_ids)
    
    # Annotate cells
    for i in range(len(task_ids)):
        for j in range(len(categories)):
            val = heatmap_arr[i, j]
            color = "white" if val < 0.4 else "black"
            ax1.text(j, i, f"{val:.0%}", ha="center", va="center", color=color, fontweight="bold")
    
    fig1.colorbar(im, ax=ax1, label="Score", shrink=0.8)
    ax1.set_title("Evaluation Heatmap — Tasks × Categories", fontweight="bold", pad=12)
    fig1.tight_layout()
    plt.show()
    
    # ── Figure 2: Per-criterion breakdown (grouped bars) ─────────────────
    fig2, axes = plt.subplots(
        len(categories), 1,
        figsize=(10, 3.5 * len(categories)),
        squeeze=False,
    )
    
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(task_ids), 3)))
    
    for cat_idx, cat in enumerate(categories):
        ax = axes[cat_idx, 0]
        criteria_keys = list(eval_criteria[cat]["criteria"].keys())
        x = np.arange(len(criteria_keys))
        width = 0.8 / max(len(task_ids), 1)
        
        for tid_idx, tid in enumerate(task_ids):
            cat_result = all_results[tid].get(cat, {})
            scores = []
            for ck in criteria_keys:
                entry = cat_result.get(ck, {})
                scores.append(entry.get("score", 0) if isinstance(entry, dict) else 0)
            
            offset = (tid_idx - len(task_ids) / 2 + 0.5) * width
            bars = ax.bar(x + offset, scores, width, label=tid, color=colors[tid_idx % len(colors)])
        
        ax.set_xticks(x)
        ax.set_xticklabels(
            [k.replace("_", " ").title() for k in criteria_keys],
            rotation=30, ha="right", fontsize=8,
        )
        ax.set_ylim(-0.1, 1.3)
        ax.set_ylabel("Score")
        ax.set_title(f"{cat.upper()} (model: {eval_criteria[cat]['model']})", fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    
    fig2.suptitle("Per-Criterion Breakdown by Task", fontweight="bold", y=1.01)
    fig2.tight_layout()
    plt.show()
    
    # ── Figure 3: Radar chart — overall category scores per task ─────────
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig3, ax3 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    for tid_idx, tid in enumerate(task_ids):
        summary = all_results[tid].get("summary", {})
        values = [summary.get(f"{cat}_score", 0) for cat in categories]
        values += values[:1]
        ax3.plot(angles, values, "o-", linewidth=2, label=tid, color=colors[tid_idx % len(colors)])
        ax3.fill(angles, values, alpha=0.1, color=colors[tid_idx % len(colors)])
    
    ax3.set_thetagrids(
        [a * 180 / np.pi for a in angles[:-1]],
        [c.title() for c in categories],
    )
    ax3.set_ylim(0, 1)
    ax3.set_title("Category Scores — Radar", fontweight="bold", pad=20)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig3.tight_layout()
    plt.show()
    
    # ── Print summary table ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Task':<25} {'Plan':>6} {'Tests':>6} {'Code':>6} {'Refl.':>6} {'Overall':>8}")
    print("-" * 70)
    for tid in task_ids:
        s = all_results[tid].get("summary", {})
        print(
            f"{tid:<25} "
            f"{s.get('plan_score', 0):>5.0%} "
            f"{s.get('tests_score', 0):>5.0%} "
            f"{s.get('code_score', 0):>5.0%} "
            f"{s.get('reflection_score', 0):>5.0%} "
            f"{s.get('overall_score', 0):>7.0%}"
        )
    print("=" * 70)


print("Visualization function defined.")