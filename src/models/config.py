import glob
import logging
import os
import pickle
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
from dataclasses import dataclass

import dotenv
import openai
from pydantic import BaseModel, ConfigDict, Field


# ── Pydantic Models ────────────────────────────────────────────────────────────

class TextNode(BaseModel):
    """Pydantic model for text nodes with embeddings."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique node identifier")
    embedding: List[float] = Field(default=[], description="Vector embedding for semantic search")
    model_config = ConfigDict(frozen=False)


class EvalTaskInput(BaseModel):
    """Single task input to the evaluator."""
    task_id: str
    query: str
    retrieval_context: str
    real_answer: str
    python_code: str
    llm_model_name: str = "gemma27"
    
@dataclass
class SearchResult:
    node: TextNode
    text: str
    score: float
    layer: int
    parent_id: Optional[str]

class EvalJudgeResult(BaseModel):
    """
    Holds the judge output for one task.

    `scores` is a plain dict whose keys mirror whatever the active prompt returns.
    `total_score` is the numeric sum of all score values.
    `verdict` captures an optional pass/fail string.
    """
    task_id: str
    llm_model_name: str
    query: str
    retrieval_context: str
    real_answer: str
    python_code: str
    reasoning: str
    scores: Dict[str, Any]
    total_score: float
    verdict: Optional[str] = None


# ── Model Mapping ──────────────────────────────────────────────────────────────

_MODEL_MAP = {
    "largest": "gemini-2.5-flash-lite",
    "gemma27": "gemma-3-27b-it",
    "gemma12": "gemma-3-12b-it",
    "gemma4": "gemma-3-4b-it",
    "gemma1": "gemma-3-1b-it",
}

QDRANT_PATH = "./data/.qdrant_storage"


# ── Logging ───────────────────────────────────────────────────────────────────

def build_logger() -> None:
    """Configure the root logger with a file handler and a stream handler."""
    os.makedirs("logs", exist_ok=True)
    _fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _file_handler = logging.FileHandler("logs/logs.log", mode="a", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(_fmt)
    _stream_handler = logging.StreamHandler()
    _stream_handler.setLevel(logging.INFO)
    _stream_handler.setFormatter(_fmt)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[_file_handler, _stream_handler],
        force=True,
    )


# ── Setup Function ─────────────────────────────────────────────────────────────

def setup() -> None:
    """Load .env, configure OpenAI client, validate required environment variables, and set up logging."""
    build_logger()
    dotenv.load_dotenv()
    openai.api_key = os.getenv("GOOGLE_API_KEY")
    openai.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    openai.max_retries = 25
    missing = [k for k in ("GOOGLE_API_KEY",) if not os.getenv(k)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {missing}")

setup()


# ── App-Level Runtime Config ─────────────────────────────────────────────────

from .schema import AgentParams, CliParams, PipelineParams, RagBuildConfig, StaticParams


@dataclass(frozen=True)
class AppConfig:
    pipeline: PipelineParams
    static: StaticParams
    agent: AgentParams
    rag: RagBuildConfig


__all__ = [
    "AgentParams",
    "AppConfig",
    "CliParams",
    "EvalJudgeResult",
    "EvalTaskInput",
    "PipelineParams",
    "QDRANT_PATH",
    "RagBuildConfig",
    "SearchResult",
    "StaticParams",
    "TextNode",
    "_MODEL_MAP",
    "build_logger",
    "setup",
]
