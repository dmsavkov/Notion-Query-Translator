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
from typing import Any, Dict, List, Literal, Optional, cast
import yaml

import openai
from json_repair import repair_json
from langsmith.wrappers import wrap_openai

from .config import (
    _MODEL_MAP,
)
from .openai_utils import create_async_openai_client
from .prompts import (
    build_concision_prompt,
    build_generate_code_prompt,
    build_generate_request_plan_prompt,
    build_reflect_code_prompt,
)

logger = logging.getLogger(__name__)


def load_eval_tasks(
    evals_dir: str = "evals",
    case_type: Literal["simple", "complex", "all"] = "simple"
) -> Dict[str, Dict[str, Any]]:
    """
    Load evaluation tasks from YAML files.
    
    Args:
        evals_dir: Base evaluation directory (e.g., "evals").
        case_type: Which eval case suite to load.
                  - "simple": evals/*.yaml only (top-level)
                  - "complex": evals/complex/*.yaml only
                  - "all": both simple and complex evals
    
    Returns:
        Dict of task_id -> task_data. Returns {} if directory not found.
    """
    tasks: Dict[str, Dict[str, Any]] = {}
    
    glob_patterns = []
    if case_type in ("simple", "all"):
        glob_patterns.append(os.path.join(evals_dir, "*.yaml"))
    if case_type in ("complex", "all"):
        glob_patterns.append(os.path.join(evals_dir, "complex", "*.yaml"))
    
    for pattern in glob_patterns:
        for yaml_path in sorted(glob.glob(pattern)):
            stem = Path(yaml_path).stem
            if stem in tasks:
                raise ValueError(
                    f"Duplicate eval task_id '{stem}' detected while loading '{yaml_path}'. "
                    "Task IDs must be unique across selected eval suites."
                )
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


def parse_statements_response(response_content: Any) -> List[Dict[str, Any]]:
    """
    Parse and repair statement-evaluator responses with minimal extraction logic.

    Supported input forms:
    - Raw model text (including fenced ```json blocks)
    - Already parsed Python list/dict

    Parsing strategy is intentionally minimal:
    - Find first '[' and last ']'
    - Repair JSON via json_repair
    - Parse list
    - Keep only items that include a statement key and a presence key
      (supports both 'present' and legacy 'status').
    """
    try:
        if response_content is None:
            return []

        parsed: Any
        if isinstance(response_content, list):
            parsed = response_content
        elif isinstance(response_content, dict):
            parsed = response_content.get("statements", [])
        else:
            text = str(response_content)
            left_idx = text.find("[")
            right_idx = text.rfind("]")
            if left_idx < 0 or right_idx <= left_idx:
                return []

            candidate = text[left_idx:right_idx + 1]
            repaired = repair_json(candidate)
            parsed = json.loads(repaired)

        if not isinstance(parsed, list):
            return []

        filtered: List[Dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue

            has_statement = "statement" in item
            has_present_field = "status" in item
            if has_statement and has_present_field:
                filtered.append(item)

        return filtered

    except Exception as e:
        logger.warning("parse_statements_response failed: %s", e)
        return []





# ── Pipeline ─────────────────────────────────────────────────────────────────────────────
_async_client = create_async_openai_client(
    api_key=os.getenv("GOOGLE_API_KEY") or "",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    max_retries=25,
)

def _check_finish_reason(model_name: str, finish_reason: str) -> None:
    """Check and log the finish_reason from LLM response."""
    print(f"[async_chat_wrapper] Model: {model_name}, finish_reason: {finish_reason}")
    if finish_reason != "stop":
        logger.warning(f"Non-stop finish_reason: {finish_reason} (response may be truncated)")


def _extract_message_content_or_raise(response: Any, model_name: str) -> str:
    """Validate chat completion payload and return non-empty text content."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"LLM response has no choices for model '{model_name}'.")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"LLM response choice has no message for model '{model_name}'.")

    content = getattr(message, "content", None)
    if content is None:
        raise ValueError(
            f"LLM response content is None for model '{model_name}' "
            "(possible safety block, refusal, or transport issue)."
        )

    content_text = content if isinstance(content, str) else str(content)
    if not content_text.strip():
        raise ValueError(f"LLM response content is empty for model '{model_name}'.")

    return content_text

async def async_chat_wrapper(
    messages: list[Dict[str, str]],
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
    
    # Add concision instruction to encourage self-restriction rather than hard cutoff
    if max_tokens and max_tokens > 0:
        concision_instruction = build_concision_prompt(max_tokens)
        msgs.append({"role": "user", "content": concision_instruction})
    
    if json_output:
        if 'gemini' in model_name:
            response = await _async_client.chat.completions.parse(
                model=model_name,
                messages=cast(Any, msgs),
                temperature=temperature,
                response_format=cast(Any, {"type": "json_object"})
            )
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise ValueError(f"LLM parse response has no choices for model '{model_name}'.")
            _check_finish_reason(model_name, str(getattr(choices[0], "finish_reason", "")))
            content = _extract_message_content_or_raise(response, model_name)
            logger.debug(f"LLM response: {content}")
            return json.loads(content)
        
        else:
            msgs.append({"role": "user", "content": "Please provide the output in JSON format."})
         
    response = await _async_client.chat.completions.create(
        model=model_name,
        messages=cast(Any, msgs),
        temperature=temperature,
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError(f"LLM response has no choices for model '{model_name}'.")
    _check_finish_reason(model_name, str(getattr(choices[0], "finish_reason", "")))
    
    content = _extract_message_content_or_raise(response, model_name)
    logger.debug(f"LLM response: {content}")
    if json_output:
        content = extract_json_from_response(content)
    return content
# ── Step 1.5: Requirements Analysis ────────────────────────────────────────────────


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


async def generate_request_plan(user_prompt: str, rag_context: str, chat_fn: partial) -> str:
    """Bullet-point plan of what needs to be implemented — large model.
    
    Args:
        user_prompt: The user's query/request.
        rag_context: Retrieved context from RAG pipeline.
        chat_fn: Optional partial function for async_chat_wrapper with pre-configured model/temperature/max_tokens.
                If None, uses async_chat_wrapper with hardcoded defaults.
    """
    prompt = build_generate_request_plan_prompt(user_prompt=user_prompt, rag_context=rag_context)
    
    return await chat_fn(
        [{"role": "user", "content": prompt}],
        json_output=False,
    )


# ── Step 4: Generate Tests ────────────────────────────────────────────────────────────────







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






print("Evaluation functions defined.")




print("Visualization function defined.")