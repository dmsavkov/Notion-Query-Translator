import json
import logging
from functools import lru_cache
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from .config import CONFIG, SearchResult, TextNode
from .prompts import (
    build_cot_decompose_prompt,
    build_domain_decompose_prompt,
    build_multi_query_prompt,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_fastembed_model() -> TextEmbedding:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info("Loading fastembed model: %s", model_name)
    return TextEmbedding(model_name=model_name)


def embed_text(text: str):
    """Embed text with fastembed and return a dense vector (numpy array)."""
    model = _get_fastembed_model()
    vector = next(iter(model.embed([text])))
    return vector


def build_qdrant_client(path: str) -> QdrantClient:
    return QdrantClient(path=path)


async def query_qdrant(
    client: QdrantClient,
    query: str,
    collection_name: str = "notion_docs",
    top_k: int = 5,
    threshold: Optional[float] = None,
) -> List[SearchResult]:
    """Search Qdrant and map scored points to SearchResult records."""
    query_vec = embed_text(query).tolist()
    response = client.query_points(
        collection_name=collection_name,
        query=query_vec,
        limit=top_k,
        score_threshold=threshold,
        with_payload=True,
    )

    points = response.points if hasattr(response, "points") else []
    results: List[SearchResult] = []

    for point in points:
        payload = point.payload or {}
        node_id = str(payload.get("node_id") or point.id)
        text = str(payload.get("text") or "")
        parent_id = payload.get("parent_id")
        layer = int(payload.get("layer") or 1)

        results.append(
            SearchResult(
                node=TextNode(id=node_id, embedding=[]),
                text=text,
                score=float(point.score or 0.0),
                layer=layer,
                parent_id=str(parent_id) if parent_id is not None else None,
            )
        )

    return results


async def summarize_retrieval_results(
    results: List[SearchResult],
    chat_fn: Callable[..., Awaitable[Any]],
) -> str:
    """Summarize retrieved chunks with concise coding-focused bullets."""
    concatenated = "\n\n".join(r.text for r in results)
    summary = await chat_fn(
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize the following retrieved context in concise bullet points to guide coding:\n\n"
                    f"{concatenated}"
                ),
            }
        ],
        max_tokens=1500,
    )
    return str(summary)


class QueryEngineer:
    """Query-engineering helpers with configurable model and temperature."""

    def __init__(
        self,
        chat_fn: Callable[..., Awaitable[Any]],
        temperature: Optional[float] = None,
        model_size: Optional[str] = None,
    ):
        self.chat_fn = chat_fn
        self.temperature = temperature if temperature is not None else CONFIG.query_engineer_temperature
        self.model_size = model_size if model_size is not None else CONFIG.default_model

    async def multi_query(self, query: str, n: Optional[int] = None) -> List[str]:
        if n is None:
            n = CONFIG.multi_query_variants

        prompt = build_multi_query_prompt(query=query, n=n)
        result = await self.chat_fn(
            messages=[{"role": "user", "content": prompt}],
            json_output=True,
            max_tokens=CONFIG.multi_query_max_tokens,
            temperature=self.temperature,
            model_size=self.model_size,
        )
        return result.get("queries", []) if isinstance(result, dict) else []

    async def cot_decompose(self, query: str) -> List[str]:
        prompt = build_cot_decompose_prompt(query=query)
        result = await self.chat_fn(
            messages=[{"role": "user", "content": prompt}],
            json_output=True,
            max_tokens=CONFIG.decomp_max_tokens,
            temperature=self.temperature,
            model_size=self.model_size,
        )
        return result.get("sub_questions", []) if isinstance(result, dict) else []

    async def domain_decompose(self, query: str) -> List[str]:
        prompt = build_domain_decompose_prompt(query=query)
        result = await self.chat_fn(
            messages=[{"role": "user", "content": prompt}],
            json_output=True,
            max_tokens=CONFIG.decomp_max_tokens,
            temperature=self.temperature,
            model_size=self.model_size,
        )
        return result.get("queries", []) if isinstance(result, dict) else []


async def search_multiple_queries(
    queries: List[str],
    search_fn: Callable[[str], Awaitable[List[SearchResult]]],
) -> List[SearchResult]:
    """Run multiple retrieval queries and deduplicate by node id."""
    all_results: List[SearchResult] = []
    for query in queries:
        all_results.extend(await search_fn(query))

    unique_nodes: Dict[str, SearchResult] = {}
    for result in all_results:
        node_id = result.node.id
        if node_id not in unique_nodes or result.score > unique_nodes[node_id].score:
            unique_nodes[node_id] = result

    consolidated = list(unique_nodes.values())
    consolidated.sort(key=lambda item: item.score, reverse=True)

    payload_entries = [
        {
            "rank": rank,
            "node_id": r.node.id,
            "score": round(r.score, 6),
            "layer": r.layer,
            "parent_id": r.parent_id,
            "text": r.text,
        }
        for rank, r in enumerate(consolidated, start=1)
    ]
    logger.info(
        "search_multiple_queries | queries=%s | total_raw=%d | consolidated=%d\n%s",
        queries,
        len(all_results),
        len(consolidated),
        json.dumps(payload_entries, ensure_ascii=False, indent=2),
    )

    return consolidated
