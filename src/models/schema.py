import operator
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field


TerminalStatus = Literal[
    "pending",
    "success",
    "security_blocked",
    "execution_failed",
    "max_retries_exceeded",
]


@dataclass(frozen=True)
class RagBuildConfig:
    corpora_path: str = "./data/corpora.txt"
    qdrant_path: str = "./data/.qdrant_storage"
    leaf_collection_name: str = "notion_docs_leaf"
    parent_collection_name: str = "notion_docs_parent"
    parent_chunk_size: int = 2000
    leaf_chunk_size: int = 200


class CodeGeneratorParams(BaseModel):
    model_name: str = "gemini-3.1-flash-lite-preview"
    model_temperature: float = 0.3
    max_tokens: Optional[int] = None

    model_config = ConfigDict(frozen=True)


class ReflectorParams(BaseModel):
    model_name: str = "gemma27"
    model_temperature: float = 0.2
    max_tokens: Optional[int] = None

    model_config = ConfigDict(frozen=True)


class QueryTranslatorParams(BaseModel):
    model_name: str = "gemma4"
    model_temperature: float = 0.3
    max_tokens: Optional[int] = None
    n_queries: int = 3
    top_k: int = 3
    top_k_total: int = 5
    query_method: Literal["multi_query", "cot_decompose", "domain_decompose"] = "cot_decompose"
    use_summarization: bool = True
    summarization_model_name: str = "gemma27"
    summarization_temperature: float = 0.2
    summarization_max_tokens: int = 200

    model_config = ConfigDict(frozen=True)


class RequestPlannerParams(BaseModel):
    model_name: str = "gemma27"
    model_temperature: float = 0.3
    max_tokens: Optional[int] = None

    model_config = ConfigDict(frozen=True)


class PrecheckGeneralParams(BaseModel):
    model_name: str = "gemma4"
    model_temperature: float = 0.0
    max_tokens: Optional[int] = None

    model_config = ConfigDict(frozen=True)


class PrecheckSecurityParams(BaseModel):
    model_name: str = "meta-llama/llama-guard-4-12b"
    base_url: str = "https://api.puter.com/puterai/openai/v1/"
    api_key_env: str = "POETRY_API_KEY"
    max_tokens: Optional[int] = None

    model_config = ConfigDict(frozen=True)


class PrecheckParams(BaseModel):
    enabled: bool = True
    general: PrecheckGeneralParams = PrecheckGeneralParams()
    security: PrecheckSecurityParams = PrecheckSecurityParams()

    model_config = ConfigDict(frozen=True)


class AgentParams(BaseModel):
    """Per-node model and retrieval settings."""

    code_generator: CodeGeneratorParams = CodeGeneratorParams()
    reflector: ReflectorParams = ReflectorParams()
    query_translator: QueryTranslatorParams = QueryTranslatorParams()
    request_planner: RequestPlannerParams = RequestPlannerParams()
    precheck: PrecheckParams = PrecheckParams()

    model_config = ConfigDict(frozen=True)


class PipelineParams(BaseModel):
    """Dynamic parameters used during pipeline execution."""

    minimal: bool = False
    max_trials: int = 3
    execution_method: Literal["local", "sandbox"] = "sandbox"
    sandbox_template: str = "notion-query-execution-sandbox"
    sandbox_client_timeout_seconds: int = 5 * 60
    sandbox_execution_timeout_seconds: int = 15
    egress_checked_tokens: List[str] = Field(default_factory=lambda: ["NOTION_TOKEN"])

    model_config = ConfigDict(frozen=True)


class CliParams(BaseModel):
    """CLI-level parameters passed into the pipeline entrypoint."""

    user_prompt: str
    think: bool = False

    model_config = ConfigDict(frozen=True)


class StaticParams(BaseModel):
    """Static parameters for pipeline initialization."""

    evals_dir: str = "evals"
    output_dir: str = "evaluation_results"
    sqlite_saver_path: str = "data/checkpoints.sqlite"
    case_type: Literal["simple", "complex", "all"] = "complex"
    context_used: str = "database_schema_report_comprehensive__notion_api_top25_20220628"
    enable_planning: bool = False
    max_concurrency: int = 6

    model_config = ConfigDict(frozen=True)


class PipelineState(TypedDict):
    task_id: str
    user_prompt: str
    meta: Dict[str, Any]
    security: Dict[str, Any]
    retrieval_context: str
    request_plan: str
    general_info: str
    trial_num: int
    generated_code: str
    function_name: str
    solution_run: Dict[str, Any]
    execution_output: str
    reflection_context: List[str]
    feedback: str
    verdict: Dict[str, Any]
    trials: List[Dict[str, Any]]
    final_code: str
    terminal_status: TerminalStatus
    queries: Annotated[List[str], operator.add]
    sandbox_id: Optional[str]
    affected_notion_ids: List[str]


def build_cli_eval_tasks(cli_params: CliParams) -> Dict[str, Dict[str, Any]]:
    return {
        "user_request": {
            "query": cli_params.user_prompt,
            "think": cli_params.think,
        }
    }


def generate_default_state() -> PipelineState:
    return {
        "task_id": "",
        "user_prompt": "",
        "meta": {},
        "security": {},
        "retrieval_context": "",
        "request_plan": "",
        "general_info": "",
        "trial_num": 0,
        "generated_code": "",
        "function_name": "",
        "solution_run": {},
        "execution_output": "",
        "reflection_context": [],
        "feedback": "",
        "verdict": {},
        "trials": [],
        "final_code": "",
        "terminal_status": "pending",
        "queries": [],
        "sandbox_id": None,
        "affected_notion_ids": [],
    }


__all__ = [
    "AgentParams",
    "CliParams",
    "PipelineParams",
    "PipelineState",
    "RagBuildConfig",
    "StaticParams",
    "TerminalStatus",
    "build_cli_eval_tasks",
    "generate_default_state",
]
