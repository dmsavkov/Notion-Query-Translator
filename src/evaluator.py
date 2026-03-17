"""
Evaluator class for RAG pipeline artifacts.

Key design:
- Every function is async
- Every function takes judge_model_name parameter
- Every function returns un-jsoned dict from LLM calls
- eval_context_statements() is generic (works for RAG, code, tests, plan context)
"""

import asyncio
import contextlib
import io
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .all_functionality import async_chat_wrapper, parse_statements_response
from .prompts import build_prompt_statements

logger = logging.getLogger(__name__)


@dataclass
class EvalInputs:
    """Container for all evaluation context fields."""
    rag_context: Optional[str] = None
    code: Optional[str] = None
    tests: Optional[str] = None
    reflection: Optional[str] = None
    plan: Optional[str] = None


class Evaluator:
    """
    Async evaluator for code pipeline artifacts.
    
    Each evaluation function:
    - Takes judge_model_name to swap models per call
    - Returns raw un-jsoned dict from LLM
    - Is async for concurrent orchestration
    """
    
    def __init__(
        self,
        default_judge_model: str = "gemma27",
        eval_temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        """
        Initialize evaluator with default model, temperature, and max tokens.
        
        Args:
            default_judge_model: Default model size for LLM calls (e.g. "gemma27", "gemini-3.1-flash")
            eval_temperature: Temperature for deterministic evaluation (default 0.1)
            max_tokens: Maximum tokens for LLM responses (default 2000)
        """
        self.default_judge_model = default_judge_model
        self.eval_temperature = eval_temperature
        self.max_tokens = max_tokens
    
    async def judge_general(
        self,
        eval_inputs: EvalInputs,
        judge_model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Holistic LLM judgment of code quality across all evaluation inputs.
        
        STUB FOR INTERFACE — to be completed later.
        
        Args:
            eval_inputs: EvalInputs with rag_context, code, tests, reflection, plan
            judge_model_name: Model to use (None = use default)
        
        Returns:
            Un-jsoned dict from LLM with judgment results
        """
        model = judge_model_name or self.default_judge_model
        
        # TODO: Build assembled context prompt, call LLM once with all non-None fields
        # For now: placeholder
        logger.info("judge_general() called but not yet implemented")
        
        return {
            "status": "stub",
            "message": "judge_general() implementation pending",
        }
    
    async def eval_context_statements(
        self,
        context: str,
        statements: List[str],
        judge_model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate statements against a given context via LLM.
        
        GENERIC validator — works for RAG context, code, tests, plans, etc.
        
        Args:
            context: Text context to validate against (can be code, RAG text, plan, etc.)
            statements: List of technical statements/requirements to verify
            judge_model_name: Model to use (None = use default)
        
        Returns:
            Un-jsoned dict with statement validation results:
            [
                {
                    "statement": "...",
                    "status": "Present | Wrong | Not present",
                    "evidence": "...",
                    "reasoning": "..."
                },
                ...
            ]
        """
        model = judge_model_name or self.default_judge_model
        
        if not statements:
            logger.warning("eval_context_statements: no statements provided")
            return []
        
        # Build prompt using reusable function
        prompt = build_prompt_statements(context, statements)
        
        # LLM call
        result = await async_chat_wrapper(
            messages=[{"role": "user", "content": prompt}],
            json_output=False,
            temperature=self.eval_temperature,
            model_size=model,
            max_tokens=self.max_tokens,
        )

        return parse_statements_response(result)
    
    async def eval_code_exec(self, code: str) -> Dict[str, Any]:
        """
        Execute code and capture success/failure, output, errors.
        
        No LLM involved — deterministic execution result.
        
        Args:
            code: Python code string to execute
        
        Returns:
            Dict with execution result:
            {
                "pass": bool,
                "output": str (stdout),
                "errors": str or None (stderr/exception),
            }
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code, {"__name__": "__main__"})
            
            result = {
                "pass": True,
                "output": stdout_buf.getvalue(),
                "errors": None,
            }
        
        except (Exception, SystemExit) as e:
            result = {
                "pass": False,
                "output": stdout_buf.getvalue(),
                "errors": str(e),
            }
        
        logger.info("eval_code_exec | pass=%s", result["pass"])
        return result
    
    async def eval_code(
        self,
        code: str,
        statements: List[str],
        judge_model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrate code execution + statement validation.
        
        Runs both in parallel (independent), then merges results.
        
        Args:
            code: Python code to evaluate
            statements: Technical statements/requirements the code should satisfy
            judge_model_name: Model for statement validation (None = use default)
        
        Returns:
            Merged dict:
            {
                "execution": {
                    "pass": bool,
                    "output": str,
                    "errors": str or None
                },
                "statements": {
                    "statements": [
                        {"statement": "...", "status": "...", "evidence": "...", ...},
                        ...
                    ]
                }
            }
        """
        # Run both evaluations in parallel (independent)
        execution_result, statements_result = await asyncio.gather(
            self.eval_code_exec(code),
            self.eval_context_statements(
                context=code,
                statements=statements,
                judge_model_name=judge_model_name,
            ),
        )
        
        # Merge with nested keys to avoid collisions
        return {
            "execution": execution_result,
            "statements": statements_result,
        }
