"""
Task 5 - Semantic search module.

Dense retrieval uses the FAISS vector store created by Task 4 in
data/vectorstore/. Query embeddings use the same all-MiniLM-L6-v2 model.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).parent.parent
VECTORSTORE_DIR = PROJECT_DIR / "data" / "vectorstore"
FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss.index"
METADATA_PATH = VECTORSTORE_DIR / "metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)


@lru_cache(maxsize=1)
def _load_faiss_index():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_INDEX_PATH}. "
            "Run src/task4_chunking_indexing.py first."
        )
    return faiss.read_index(str(FAISS_INDEX_PATH))


@lru_cache(maxsize=1)
def _load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"FAISS metadata not found: {METADATA_PATH}. "
            "Run src/task4_chunking_indexing.py first."
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _embed_query(query: str) -> np.ndarray:
    model = _load_model()
    embedding = model.encode(query, normalize_embeddings=True)
    vector = np.array([embedding], dtype="float32")
    faiss.normalize_L2(vector)
    return vector


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search semantically using FAISS cosine similarity.

    Args:
        query: User query.
        top_k: Maximum number of results.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted by
        score descending.
    """
    if top_k <= 0:
        return []

    query = query.strip()
    if not query:
        return []

    index = _load_faiss_index()
    metadata = _load_metadata()
    chunks = metadata.get("chunks", [])
    if index.ntotal == 0 or not chunks:
        return []

    limit = min(top_k, index.ntotal)
    scores, indices = index.search(_embed_query(query), limit)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[int(idx)]
        results.append(
            {
                "content": chunk.get("content", ""),
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    results = semantic_search("hình phạt cho tội tàng trữ ma túy", top_k=5)
    for result in results:
        source = result.get("metadata", {}).get("source", "unknown")
        print(f"[{result['score']:.3f}] {source}: {result['content'][:120]}...")
