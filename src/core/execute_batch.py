import asyncio
from typing import Any, Dict

from ..models.config import AppConfig
from ..utils.execution_utils import generate_thread_id
from .execute_single import execute_single


async def execute_batch(
    *,
    tasks: Dict[str, Dict[str, Any]],
    app_config: AppConfig,
    pipeline: Any,
    qdrant_client: Any = None,
) -> Dict[str, Dict[str, Any]]:
    semaphore = asyncio.Semaphore(app_config.static.max_concurrency)

    async def run_task(task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            return await execute_single(
                tasks={task_id: task_data},
                app_config=app_config,
                pipeline=pipeline,
                thread_id=generate_thread_id(task_id),
                qdrant_client=qdrant_client,
            )

    task_results = await asyncio.gather(
        *(run_task(task_id, task_data) for task_id, task_data in tasks.items())
    )
    merged_results: Dict[str, Dict[str, Any]] = {}
    for task_result in task_results:
        merged_results.update(task_result)
    return merged_results
