"""
Task 10 - Generation with citations.

LLM provider:
    Alibaba Cloud DashScope / Model Studio OpenAI-compatible API.

Required .env values for real generation:
    DASHSCOPE_API_KEY=...
    DASHSCOPE_BASE_URL=https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
    DASHSCOPE_MODEL=qwen3.5-flash

If the DashScope config is missing, the module falls back to an extractive answer
from retrieved context so tests and local demos still run without an API call.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve

PROJECT_DIR = Path(__file__).parent.parent
load_dotenv(PROJECT_DIR / ".env")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "").strip()
DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen3.5-flash").strip()

# top_k=5 gives enough evidence diversity without overloading a flash model.
DEFAULT_TOP_K = 5

# top_p=0.3 keeps the answer grounded and less creative for legal/news QA.
DEFAULT_TOP_P = 0.3

SYSTEM_PROMPT = """Answer the question using only the provided context.
For every factual claim, immediately add a citation in brackets like [Source, Year].
If the provided context does not explicitly support the answer, say "I cannot verify this information".
Do not invent citations."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to reduce lost-in-the-middle.

    The highest ranked chunk stays first. The next important chunks alternate
    between the end and the middle, keeping strong evidence near both edges.
    """
    if len(chunks) <= 2:
        return chunks[:]

    reordered: list[dict] = [chunks[0]]
    tail: list[dict] = []

    for index, chunk in enumerate(chunks[1:], start=1):
        if index % 2 == 1:
            tail.insert(0, chunk)
        else:
            reordered.append(chunk)

    return reordered + tail


def _source_year(metadata: dict) -> tuple[str, str]:
    source = (
        metadata.get("source")
        or metadata.get("path")
        or metadata.get("filename")
        or "Unknown source"
    )
    year = "2026"
    match = re.search(r"(20\d{2})", str(source))
    if match:
        year = match.group(1)
    return str(source), year


def format_context(chunks: list[dict]) -> str:
    """
    Format context chunks with source metadata for citation.
    """
    formatted = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source, year = _source_year(metadata)
        formatted.append(
            f"[Context {index}]\n"
            f"Source: {source}\n"
            f"Year: {year}\n"
            f"Score: {float(chunk.get('score', 0.0)):.4f}\n"
            f"Content:\n{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(formatted)


def _dashscope_client() -> OpenAI | None:
    if not DASHSCOPE_API_KEY or not DASHSCOPE_BASE_URL:
        return None
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


def _fallback_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I cannot verify this information"

    best = chunks[0]
    source, year = _source_year(best.get("metadata", {}))
    content = " ".join(str(best.get("content", "")).split())
    excerpt = content[:700].rstrip()
    if not excerpt:
        return "I cannot verify this information"

    return (
        f"Based on the retrieved context, the most relevant evidence says: "
        f"{excerpt} [{source}, {year}]."
    )


def generate_with_citation(
    query: str,
    context_chunks: list[dict] | None = None,
    top_k: int = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
) -> dict:
    """
    Generate an answer with citations.

    Steps:
        1. Retrieve context if context_chunks is not provided.
        2. Reorder context to reduce lost-in-the-middle.
        3. Format source metadata for citations.
        4. Call Alibaba DashScope OpenAI-compatible API using qwen3.5-flash.
        5. Return {'answer': str, 'sources': list, 'model': str}.
    """
    if context_chunks is None:
        context_chunks = retrieve(query, top_k=top_k)

    ordered_chunks = reorder_for_llm(context_chunks[:top_k])
    context = format_context(ordered_chunks)
    client = _dashscope_client()

    if client is None:
        return {
            "answer": _fallback_answer(query, ordered_chunks),
            "sources": ordered_chunks,
            "model": "extractive_fallback",
        }

    user_prompt = f"""Question:
{query}

Context:
{context}

Write a concise Vietnamese answer with citations. If evidence is insufficient,
write exactly: I cannot verify this information"""

    try:
        completion = client.chat.completions.create(
            model=DASHSCOPE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            top_p=top_p,
            temperature=0.2,
        )
        answer = completion.choices[0].message.content or ""
    except Exception as exc:
        answer = _fallback_answer(query, ordered_chunks)
        return {
            "answer": answer,
            "sources": ordered_chunks,
            "model": "extractive_fallback",
            "error": str(exc),
        }

    return {
        "answer": answer.strip() or "I cannot verify this information",
        "sources": ordered_chunks,
        "model": DASHSCOPE_MODEL,
    }


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    result = generate_with_citation("Hình phạt cho tội tàng trữ trái phép chất ma túy là gì?")
    print(result["answer"])
