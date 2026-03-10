import json
import logging
from functools import lru_cache
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from .config import QDRANT_PATH, SearchResult, TextNode
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


# Global Qdrant client instance
qdrant_client = build_qdrant_client(QDRANT_PATH)


async def query_qdrant(
    query: str,
    collection_name: str = "notion_docs_leaf",
    top_k: int = 5,
    threshold: Optional[float] = None,
    parent_collection_name: str = "notion_docs_parent",
    use_parent_texts: bool = True,
) -> List[SearchResult]:
    """Search Qdrant and map scored points to SearchResult records.
    
    If use_parent_texts is True, retrieves parent text from parent_collection_name
    and replaces the leaf text with the parent text.
    """
    query_vec = embed_text(query).tolist()
    response = qdrant_client.query_points(
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

        # If use_parent_texts and parent_id exists, retrieve parent text
        if use_parent_texts and parent_id is not None:
            try:
                parent_points = qdrant_client.retrieve(
                    collection_name=parent_collection_name,
                    ids=[parent_id],
                    with_payload=True,
                )
                if parent_points:
                    parent_payload = parent_points[0].payload or {}
                    text = str(parent_payload.get("text") or text)
            except Exception as e:
                logger.warning(
                    "Failed to retrieve parent text for parent_id=%s: %s",
                    parent_id,
                    e,
                )

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
    query: str,  # Added query to ground the relevance
    chat_fn: Callable[..., Awaitable[Any]],
) -> str:
    """
    Extracts high-density technical specifications from RAG chunks, 
    discarding prose and preserving literal API syntax.
    """
    concatenated = "\n\n".join(f"CHUNK {i}:\n{r.text}" for i, r in enumerate(results))
    
    prompt = f"""
    You are a Technical Document Parser. Your goal is to extract a High-Density Technical Specification from Notion API documentation for the following query.

    QUERY: "{query}"

    ### INSTRUCTIONS:
    1. REMOVE NOISE: Discard all introductory prose, marketing language, general explanations, and non-essential sidebars.
    2. PRESERVE LITERALS: Extract exact Endpoint URLs, HTTP Methods (POST/PATCH), and mandatory Header values (e.g., Notion-Version).
    3. SCHEMA EXTRACTION: For every property mentioned, extract the literal JSON nesting path. 
       - DO NOT summarize as "The name property." 
       - DO extract as: "Name": {{"title": [{{"text": {{"content": "..."}}}}]}}
    4. DATA TYPES: Explicitly list required types (e.g., 'number' vs 'string').

    ### OUTPUT FORMAT:
    - **Endpoint & Method:** [Literal string]
    - **Required Headers:** [Literal strings]
    - **Literal Schema Snippets:** [Code blocks of JSON structures found in text]
    - **Constraints:** [Specific validation rules like "max 2000 characters"]

    DOCUMENTATION CONTEXT:
    {concatenated}
    """

    summary = await chat_fn(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    return str(summary)


class QueryEngineer:
    """Query-engineering helpers that use a pre-configured chat function."""

    def __init__(
        self,
        chat_fn: Callable[..., Awaitable[Any]],
        n_queries: int = 4,
    ):
        self.chat_fn = chat_fn
        self.n_queries = n_queries

    async def multi_query(self, query: str, n: Optional[int] = None) -> List[str]:
        if n is None:
            n = self.n_queries

        prompt = build_multi_query_prompt(query=query, n=n)
        result = await self.chat_fn(
            messages=[{"role": "user", "content": prompt}],
            json_output=True,
        )
        return result.get("queries", []) if isinstance(result, dict) else []

    async def cot_decompose(self, query: str) -> List[str]:
        prompt = build_cot_decompose_prompt(query=query)
        result = await self.chat_fn(
            messages=[{"role": "user", "content": prompt}],
            json_output=True,
        )
        return result.get("sub_questions", []) if isinstance(result, dict) else []

    async def domain_decompose(self, query: str) -> List[str]:
        prompt = build_domain_decompose_prompt(query=query)
        result = await self.chat_fn(
            messages=[{"role": "user", "content": prompt}],
            json_output=True,
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
        "search_multiple_queries | queries=%s | total_raw=%d | consolidated=%d",
        len(queries),
        len(all_results),
        len(consolidated),
    )
    logger.debug("search_multiple_queries | queries=%s", json.dumps(queries, indent=2, ensure_ascii=False))
    logger.debug("search_multiple_queries | payload_entries=%s", json.dumps(payload_entries, indent=2, ensure_ascii=False))

    return consolidated
