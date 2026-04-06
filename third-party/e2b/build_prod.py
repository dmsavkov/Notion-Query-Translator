import asyncio
import os
from e2b import AsyncTemplate, default_build_logger
from template import template

import dotenv
dotenv.load_dotenv()

async def main():
    await AsyncTemplate.build(
        template,
        "notion-query-execution-sandbox",
        on_build_logs=default_build_logger(),
        cpu_count=1,
        memory_mb=512,
        api_key=os.getenv("E2B_API_KEY")
    )


if __name__ == "__main__":
    asyncio.run(main())