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


# ── Application Configuration ──────────────────────────────────────────────────

class AppConfig(BaseModel):
    """Essential application configuration. All other values remain hardcoded."""
    model_config = ConfigDict(frozen=True)

    default_model: str = "gemma27"
    query_engineer_temperature: float = 0.3
    multi_query_variants: int = 4
    multi_query_max_tokens: int = 400
    decomp_max_tokens: int = 400
    max_trials: int = 3


CONFIG = AppConfig()


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


# ── Data Loaders ───────────────────────────────────────────────────────────────

def load_storage(path: str = "vectors/storage.pkl") -> Dict[str, Any]:
    """Load hierarchical chunk storage from pickle."""
    resolved_path = Path(path)
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"Vector storage file not found: {resolved_path.resolve()}")
    with open(resolved_path, "rb") as f:
        return pickle.load(f)


def load_corpora_vectorized(path: str = "vectors/corpora_vectorized.pkl") -> List:
    """Load vectorized leaf nodes from pickle."""
    resolved_path = Path(path)
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"Vectorized corpora file not found: {resolved_path.resolve()}")
    with open(resolved_path, "rb") as f:
        return pickle.load(f)


def load_eval_tasks(evals_dir: str = "evals") -> Dict[str, str]:
    """Load evaluation tasks from YAML files. Returns {} if directory not found."""
    tasks: Dict[str, str] = {}
    for yaml_path in sorted(glob.glob(os.path.join(evals_dir, "*.yaml"))):
        stem = Path(yaml_path).stem
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        tasks[stem] = data.get("task", "")
    return tasks


def load_corpora_text(path: str = "corpora.txt") -> str:
    """Load raw corpora text from file. Returns '' if not found."""
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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

