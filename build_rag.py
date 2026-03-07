from dataclasses import dataclass
from typing import Dict, List
from uuid import uuid4

from chonkie import RecursiveChunker, Chunk
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from src.config import QDRANT_PATH
from src.rag_utils import embed_text


@dataclass
class RagBuildConfig:
    corpora_path: str = "./data/corpora.txt"
    qdrant_path: str = QDRANT_PATH
    leaf_collection_name: str = "notion_docs_leaf"
    parent_collection_name: str = "notion_docs_parent"
    parent_chunk_size: int = 2000
    leaf_chunk_size: int = 200


def load_corpora(cfg: RagBuildConfig) -> str:
    with open(cfg.corpora_path, "r", encoding="utf-8") as fh:
        return fh.read()


def _chunk_text(text: list[str], chunk_size: int, overlap: int = 100) -> List[List[Chunk]]:
    """Use chonkie recursive character splitter for smarter chunking."""
    splitter = RecursiveChunker(
        chunk_size=chunk_size,
        #chunk_overlap=overlap,
    )
    return splitter.chunk_batch(text)

def create_id() -> str:
    """Generate a unique ID for each chunk."""
    return str(uuid4())


def chunk_corpora(text: str, cfg: RagBuildConfig) -> Dict[str, Dict[str, str]]:
    """Build 2-layer hierarchy: parent chunks (layer 0) and leaf chunks (layer 1)."""
    storage: Dict[str, Dict[str, str]] = {}
    parent_chunks_matrix = _chunk_text([text], cfg.parent_chunk_size, overlap=200)
    parent_chunks = parent_chunks_matrix[0]
    parent_ids = []

    for parent_chunk in parent_chunks:
        parent_id = create_id()
        parent_ids.append(parent_id)
        storage[parent_id] = {
            "node_id": parent_id,
            "parent_id": "",
            "text": parent_chunk.text,
            "token_count": parent_chunk.token_count,
            "start_index": parent_chunk.start_index,
            "layer": "0",
            "is_leaf": "0",
        }

    leaf_chunks_matrix = _chunk_text([c.text for c in parent_chunks], cfg.leaf_chunk_size, overlap=50)
    for parent_id, leaf_chunks in zip(parent_ids, leaf_chunks_matrix):
        for leaf_chunk in leaf_chunks:
            leaf_id = create_id()
            storage[leaf_id] = {
                "node_id": leaf_id,
                "parent_id": parent_id,
                "text": leaf_chunk.text,
                "token_count": leaf_chunk.token_count,
                "start_index": leaf_chunk.start_index,
                "layer": "1",
                "is_leaf": "1",
            }

    return storage


def embed_and_upsert(storage: Dict[str, Dict[str, str]], cfg: RagBuildConfig) -> None:
    """Create two collections: leaf_embeddings (with vectors) and parent_docs (minimal vectors)."""
    client = QdrantClient(path=cfg.qdrant_path)
    
    if client.collection_exists(cfg.leaf_collection_name) or client.collection_exists(cfg.parent_collection_name):
        raise ValueError(f"One or both collections '{cfg.leaf_collection_name}' and '{cfg.parent_collection_name}' already exist. Please delete them before running this script.")

    # Get vector dimension from first embed
    sample_vec = embed_text("dimension probe").tolist()
    vec_dim = len(sample_vec)

    # ── Leaf Collection: Full embeddings ────────────────────────────────────
    client.create_collection(
        collection_name=cfg.leaf_collection_name,
        vectors_config=qmodels.VectorParams(size=vec_dim, distance=qmodels.Distance.COSINE),
    )

    # ── Parent Collection: Minimal vectors (all zeros for memory efficiency) ─
    client.create_collection(
        collection_name=cfg.parent_collection_name,
        vectors_config=qmodels.VectorParams(size=vec_dim, distance=qmodels.Distance.COSINE),
    )

    leaf_points: List[qmodels.PointStruct] = []
    parent_points: List[qmodels.PointStruct] = []

    for node in storage.values():
        node_id = node["node_id"]
        payload = {
            "node_id": node_id,
            "parent_id": node["parent_id"],
            "text": node["text"],
            "layer": int(node["layer"]),
            "is_leaf": bool(int(node["is_leaf"])),
        }

        if payload["is_leaf"]:
            # Leaf nodes: embed and store in leaf collection
            leaf_points.append(
                qmodels.PointStruct(
                    id=node_id,
                    vector=embed_text(node["text"]).tolist(),
                    payload=payload,
                )
            )
        else:
            # Parent nodes: use minimal [0.0] vectors for memory efficiency
            parent_points.append(
                qmodels.PointStruct(
                    id=node_id,
                    vector=[0.0] * vec_dim,
                    payload=payload,
                )
            )

    # Upsert to respective collections
    if leaf_points:
        client.upsert(collection_name=cfg.leaf_collection_name, points=leaf_points)
        print(f"Upserted {len(leaf_points)} leaf points to '{cfg.leaf_collection_name}'")
    
    if parent_points:
        client.upsert(collection_name=cfg.parent_collection_name, points=parent_points)
        print(f"Upserted {len(parent_points)} parent points to '{cfg.parent_collection_name}' (minimal vectors)")


if __name__ == "__main__":
    cfg = RagBuildConfig()
    text = load_corpora(cfg)
    storage = chunk_corpora(text, cfg)
    embed_and_upsert(storage, cfg)
    print("Done: Qdrant DB built with 2 collections (leaf embeddings + parent docs with minimal vectors)")
