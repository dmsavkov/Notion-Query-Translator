import asyncio
import warnings
from typing import Any, Dict, Optional

from notion_query.environment import initialize_runtime_environment
from src.adapters.cli_factory import build_app_config_from_cli
from src.adapters.cli_presenter import format_lifecycle_result
from src.all_functionality import load_eval_tasks
from src.core.lifecycle import run_with_lifecycle
from src.models.config import AppConfig
from src.models.schema import (
    AgentParams,
    CliParams,
    PipelineParams,
    RagBuildConfig,
    StaticParams,
    build_cli_eval_tasks,
)


async def run(
    eval_tasks: Dict[str, Dict[str, Any]],
    static_params: StaticParams,
    pipeline_params: PipelineParams,
    agent_params: AgentParams,
    rag_build_config: RagBuildConfig,
) -> Dict[str, Dict[str, Any]]:
    app_config = AppConfig(
        pipeline=pipeline_params,
        static=static_params,
        agent=agent_params,
        rag=rag_build_config,
    )
    return await run_with_lifecycle(tasks=eval_tasks, app_config=app_config)


async def main(
    static_params: Optional[StaticParams] = None,
    pipeline_params: Optional[PipelineParams] = None,
    agent_params: Optional[AgentParams] = None,
    rag_build_config: Optional[RagBuildConfig] = None,
    cli_params: Optional[CliParams] = None,
    dev_mode: bool = True,
) -> Dict[str, Dict[str, Any]]:
    initialize_runtime_environment(required_keys=("NOTION_TOKEN", "GOOGLE_API_KEY"))

    final_static_params = static_params or StaticParams()
    final_pipeline_params = pipeline_params or PipelineParams()
    final_agent_params = agent_params or AgentParams()
    final_rag_build_config = rag_build_config or RagBuildConfig()

    if cli_params is not None and dev_mode:
        warnings.warn(
            "cli_params were provided while dev_mode=True; development mode is turned off "
            "and CLI task input is used.",
            stacklevel=2,
        )

    if cli_params is not None:
        app_config = build_app_config_from_cli(
            cli_params=cli_params,
            static_params=final_static_params,
            pipeline_params=final_pipeline_params,
            agent_params=final_agent_params,
            rag_build_config=final_rag_build_config,
        )
        eval_tasks = build_cli_eval_tasks(cli_params)
        return await run_with_lifecycle(tasks=eval_tasks, app_config=app_config)

    if dev_mode:
        eval_tasks = load_eval_tasks(
            evals_dir=final_static_params.evals_dir,
            case_type=final_static_params.case_type,
        )
        return await run(
            eval_tasks=eval_tasks,
            static_params=final_static_params,
            pipeline_params=final_pipeline_params,
            agent_params=final_agent_params,
            rag_build_config=final_rag_build_config,
        )

    raise ValueError("No evaluation tasks were provided or loaded.")


if __name__ == "__main__":
    result = asyncio.run(main())
    print(format_lifecycle_result(result))
