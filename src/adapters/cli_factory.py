from typing import Optional

from ..models.config import AppConfig
from ..models.schema import AgentParams, CliParams, PipelineParams, RagBuildConfig, StaticParams


def build_app_config_from_cli(
    *,
    cli_params: CliParams,
    static_params: Optional[StaticParams] = None,
    pipeline_params: Optional[PipelineParams] = None,
    agent_params: Optional[AgentParams] = None,
    rag_build_config: Optional[RagBuildConfig] = None,
) -> AppConfig:
    final_pipeline_params = (pipeline_params or PipelineParams()).model_copy(
        update={"minimal": not cli_params.think}
    )

    return AppConfig(
        pipeline=final_pipeline_params,
        static=static_params or StaticParams(),
        agent=agent_params or AgentParams(),
        rag=rag_build_config or RagBuildConfig(),
    )
