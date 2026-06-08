"""
Task 6 - Lexical search module using BM25.

BM25 is built over the same chunk corpus stored by Task 4 in
data/vectorstore/metadata.json. It scores exact keyword matches with term
frequency, inverse document frequency, and document-length normalization.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

PROJECT_DIR = Path(__file__).parent.parent
METADATA_PATH = PROJECT_DIR / "data" / "vectorstore" / "metadata.json"


def _tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese text simply and consistently for BM25."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


@lru_cache(maxsize=1)
def load_corpus() -> list[dict]:
    """
    Load chunk corpus from Task 4 metadata.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Corpus metadata not found: {METADATA_PATH}. "
            "Run src/task4_chunking_indexing.py first."
        )

    payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    corpus = []
    for chunk in payload.get("chunks", []):
        content = chunk.get("content", "").strip()
        if not content:
            continue
        corpus.append(
            {
                "content": content,
                "metadata": chunk.get("metadata", {}),
            }
        )
    return corpus


def build_bm25_index(corpus: list[dict]):
    """
    Build a BM25 index from a chunk corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


@lru_cache(maxsize=1)
def _get_bm25_and_corpus():
    corpus = load_corpus()
    return build_bm25_index(corpus), corpus


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search exact keywords using BM25.

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

    bm25, corpus = _get_bm25_and_corpus()
    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append(
            {
                "content": corpus[int(idx)]["content"],
                "score": score,
                "metadata": corpus[int(idx)]["metadata"],
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    results = lexical_search("Điều 248 tàng trữ trái phép chất ma túy", top_k=5)
    for result in results:
        source = result.get("metadata", {}).get("source", "unknown")
        print(f"[{result['score']:.3f}] {source}: {result['content'][:120]}...")
