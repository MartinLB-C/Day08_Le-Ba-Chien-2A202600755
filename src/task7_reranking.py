"""
Task 7 - Reranking module.

Primary method: Jina Reranker v2 multilingual cross-encoder via Jina AI API.
If JINA_API_KEY is not configured, a deterministic local lexical fallback is used
so tests and offline demos still run.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).parent.parent
load_dotenv(PROJECT_DIR / ".env")

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _local_relevance_score(query: str, content: str, original_score: float = 0.0) -> float:
    """
    Offline fallback based on token overlap plus the retrieval score.

    This is not a replacement for Jina's cross-encoder; it only keeps the module
    usable when no API key is available.
    """
    query_tokens = set(_tokenize(query))
    content_tokens = set(_tokenize(content))
    if not query_tokens or not content_tokens:
        return float(original_score)

    overlap = len(query_tokens & content_tokens) / len(query_tokens)
    return float(0.75 * overlap + 0.25 * original_score)


def _fallback_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    scored = []
    for candidate in candidates:
        item = candidate.copy()
        item["score"] = _local_relevance_score(
            query,
            str(candidate.get("content", "")),
            float(candidate.get("score", 0.0)),
        )
        item["rerank_model"] = "local_lexical_fallback"
        scored.append(item)

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates using Jina Reranker v2.

    Args:
        query: User query.
        candidates: List of {'content': str, 'score': float, 'metadata': dict}.
        top_k: Number of reranked results to return.

    Returns:
        List of top_k candidates with updated 'score', sorted descending.
    """
    if top_k <= 0 or not candidates:
        return []

    api_key = os.getenv("JINA_API_KEY", "").strip()
    if not api_key or api_key == "jina_xxx":
        return _fallback_rerank(query, candidates, top_k)

    documents = [str(candidate.get("content", "")) for candidate in candidates]
    payload = {
        "model": JINA_RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(candidates)),
    }

    try:
        response = requests.post(
            JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        print(f"Jina reranker unavailable, using local fallback: {exc}")
        return _fallback_rerank(query, candidates, top_k)

    results = []
    for reranked in data.get("results", []):
        index = int(reranked["index"])
        if index < 0 or index >= len(candidates):
            continue

        item = candidates[index].copy()
        item["score"] = float(reranked.get("relevance_score", 0.0))
        item["rerank_model"] = JINA_RERANK_MODEL
        results.append(item)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """MMR is not the selected method for this task."""
    raise NotImplementedError("Task 7 uses Jina Reranker v2 cross-encoder")


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion helper for future hybrid retrieval experiments.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = str(item.get("content", ""))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items[key] = item

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    results = []
    for content, score in ranked[:top_k]:
        item = items[content].copy()
        item["score"] = float(score)
        item["rerank_model"] = "rrf"
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "rrf":
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    if method == "mmr":
        raise NotImplementedError("Task 7 uses Jina Reranker v2 cross-encoder")
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dummy_candidates = [
        {"content": "Điều 249: Tội tàng trữ trái phép chất ma túy", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ bị bắt vì sử dụng ma túy", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    for result in rerank("hình phạt tàng trữ ma túy", dummy_candidates, top_k=2):
        print(f"[{result['score']:.3f}] {result['content']}")
