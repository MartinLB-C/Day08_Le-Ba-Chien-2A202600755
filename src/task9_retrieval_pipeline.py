"""
Task 9 - Complete retrieval pipeline.

This pipeline combines:
    1. Semantic search from FAISS (Task 5)
    2. Lexical BM25 search (Task 6)
    3. Reciprocal Rank Fusion, RRF (Task 7 helper)
    4. Optional Jina reranking (Task 7)

PageIndex fallback is intentionally disabled because PageIndex credits are low.
"""

from __future__ import annotations

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
except ImportError:
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.0
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"
RRF_K = 60


def _mark_results(results: list[dict], ranker: str) -> list[dict]:
    marked = []
    for rank, item in enumerate(results, start=1):
        copied = item.copy()
        metadata = copied.get("metadata", {}).copy()
        metadata["ranker"] = ranker
        metadata["rank"] = rank
        copied["metadata"] = metadata
        marked.append(copied)
    return marked


def _ensure_hybrid_source(results: list[dict]) -> list[dict]:
    final = []
    for item in results:
        copied = item.copy()
        copied["source"] = "hybrid"
        final.append(copied)
    return final


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Run hybrid retrieval with RRF merge and no PageIndex fallback.

    Args:
        query: User query.
        top_k: Number of final results.
        score_threshold: Minimum accepted score. Results below this threshold are
            filtered, but no PageIndex fallback is called.
        use_reranking: Whether to rerank RRF results with Task 7 Jina reranker.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'hybrid'}.
    """
    if top_k <= 0:
        return []

    query = query.strip()
    if not query:
        return []

    search_k = max(top_k * 4, 10)
    dense_results = _mark_results(semantic_search(query, top_k=search_k), "semantic")
    sparse_results = _mark_results(lexical_search(query, top_k=search_k), "lexical")

    merged = rerank_rrf([dense_results, sparse_results], top_k=search_k, k=RRF_K)
    merged = _ensure_hybrid_source(merged)

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        final_results = _ensure_hybrid_source(final_results)
    else:
        final_results = merged[:top_k]

    filtered = [
        item for item in final_results
        if float(item.get("score", 0.0)) >= score_threshold
    ]
    return filtered[:top_k]


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy",
        "Nghệ sĩ nào bị bắt vì sử dụng ma túy",
        "Luật phòng chống ma túy 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, result in enumerate(results, start=1):
            print(
                f"  {i}. [{result['score']:.3f}] "
                f"[{result['source']}] {result['content'][:100]}..."
            )
