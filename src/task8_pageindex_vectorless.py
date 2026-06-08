"""
Task 8 - PageIndex vectorless retrieval.

The installed PageIndex SDK uploads PDF files, so this task indexes the original
legal PDFs from data/landing/legal/. Uploaded PageIndex document IDs are cached
in data/pageindex/documents.json and used for later retrieval queries.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pageindex import PageIndexAPIError, PageIndexClient

PROJECT_DIR = Path(__file__).parent.parent
LANDING_LEGAL_DIR = PROJECT_DIR / "data" / "landing" / "legal"
PAGEINDEX_DIR = PROJECT_DIR / "data" / "pageindex"
DOC_CACHE_PATH = PAGEINDEX_DIR / "documents.json"

load_dotenv(PROJECT_DIR / ".env")
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
PAGEINDEX_ALLOW_UPLOADS = os.getenv("PAGEINDEX_ALLOW_UPLOADS", "").strip() == "1"
PAGEINDEX_ALLOW_QUERIES = os.getenv("PAGEINDEX_ALLOW_QUERIES", "").strip() == "1"


def _client() -> PageIndexClient:
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY == "pi_xxx":
        raise RuntimeError("PAGEINDEX_API_KEY is missing. Add it to .env first.")
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_doc_cache() -> dict[str, dict]:
    if not DOC_CACHE_PATH.exists():
        return {}
    return json.loads(DOC_CACHE_PATH.read_text(encoding="utf-8"))


def _save_doc_cache(cache: dict[str, dict]) -> None:
    PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
    DOC_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _legal_pdfs() -> list[Path]:
    return sorted(
        file
        for file in LANDING_LEGAL_DIR.iterdir()
        if file.is_file() and file.suffix.lower() == ".pdf"
    )


def upload_documents(wait_until_ready: bool = False, poll_seconds: int = 5, timeout_seconds: int = 180) -> list[dict]:
    """
    Upload legal PDF documents to PageIndex and cache their doc IDs.

    Args:
        wait_until_ready: Poll until PageIndex reports retrieval_ready.
        poll_seconds: Delay between readiness checks.
        timeout_seconds: Max wait per newly uploaded document.

    Returns:
        Cached document records.
    """
    if not PAGEINDEX_ALLOW_UPLOADS:
        cache = _load_doc_cache()
        print(
            "PageIndex upload is disabled to protect credits. "
            "Set PAGEINDEX_ALLOW_UPLOADS=1 in .env to upload more documents."
        )
        return list(cache.values())

    client = _client()
    cache = _load_doc_cache()

    for pdf_file in _legal_pdfs():
        cache_key = pdf_file.name
        if cache_key in cache and cache[cache_key].get("doc_id"):
            print(f"Already uploaded: {pdf_file.name} -> {cache[cache_key]['doc_id']}")
            continue

        print(f"Uploading: {pdf_file.name}")
        try:
            result = client.submit_document(str(pdf_file))
        except PageIndexAPIError as exc:
            if "LimitReached" in str(exc) and cache:
                print(f"  Upload limit reached; using {len(cache)} cached PageIndex documents.")
                break
            raise
        doc_id = result.get("doc_id") or result.get("id")
        if not doc_id:
            raise PageIndexAPIError(f"Upload response did not include doc_id: {result}")

        cache[cache_key] = {
            "doc_id": doc_id,
            "filename": pdf_file.name,
            "path": pdf_file.relative_to(PROJECT_DIR).as_posix(),
            "type": "legal",
        }
        _save_doc_cache(cache)
        print(f"  Uploaded: {pdf_file.name} -> {doc_id}")

        if wait_until_ready:
            deadline = time.time() + timeout_seconds
            while time.time() < deadline:
                if client.is_retrieval_ready(doc_id):
                    print(f"  Retrieval ready: {doc_id}")
                    break
                time.sleep(poll_seconds)

    return list(cache.values())


def _extract_text_items(value: Any) -> list[str]:
    """Best-effort extraction for PageIndex retrieval response shapes."""
    texts: list[str] = []

    if isinstance(value, str):
        text = value.strip()
        if text:
            texts.append(text)
    elif isinstance(value, list):
        for item in value:
            texts.extend(_extract_text_items(item))
    elif isinstance(value, dict):
        for key in (
            "text",
            "content",
            "markdown",
            "snippet",
            "answer",
            "relevant_content",
            "section_title",
        ):
            if key in value:
                texts.extend(_extract_text_items(value[key]))
        for key in (
            "results",
            "chunks",
            "nodes",
            "retrieval",
            "references",
            "citations",
            "retrieved_nodes",
            "relevant_contents",
        ):
            if key in value:
                texts.extend(_extract_text_items(value[key]))

    return texts


def _submit_and_poll_query(client: PageIndexClient, doc_id: str, query: str, timeout_seconds: int = 120) -> dict:
    submitted = client.submit_query(doc_id=doc_id, query=query, thinking=False)
    retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
    if not retrieval_id:
        raise PageIndexAPIError(f"Query response did not include retrieval_id: {submitted}")

    deadline = time.time() + timeout_seconds
    last_result: dict = {}
    while time.time() < deadline:
        last_result = client.get_retrieval(retrieval_id)
        status = str(last_result.get("status", "")).lower()
        if status in {"completed", "complete", "done", "success", "succeeded"}:
            return last_result
        if status in {"failed", "error"}:
            raise PageIndexAPIError(f"Retrieval failed: {last_result}")
        if _extract_text_items(last_result):
            return last_result
        time.sleep(3)

    return last_result


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
    """
    if top_k <= 0:
        return []

    query = query.strip()
    if not query:
        return []

    client = _client()
    cache = _load_doc_cache()
    if not cache:
        print("No cached PageIndex doc_id found. Upload is disabled by default to protect credits.")
        return []

    if not PAGEINDEX_ALLOW_QUERIES:
        print(
            "PageIndex query is disabled to protect credits. "
            "Set PAGEINDEX_ALLOW_QUERIES=1 in .env to run retrieval."
        )
        return []

    results: list[dict] = []
    per_doc_limit = max(1, top_k)
    for record in cache.values():
        if len(results) >= top_k:
            break

        try:
            retrieval = _submit_and_poll_query(client, record["doc_id"], query)
        except Exception as exc:
            print(f"PageIndex query failed for {record.get('filename')}: {exc}")
            continue

        texts = _extract_text_items(retrieval)
        for rank, text in enumerate(texts[:per_doc_limit], start=1):
            results.append(
                {
                    "content": text,
                    "score": 1.0 / rank,
                    "metadata": {
                        "doc_id": record.get("doc_id"),
                        "source": record.get("filename"),
                        "path": record.get("path"),
                        "type": record.get("type", "legal"),
                    },
                    "source": "pageindex",
                }
            )
            if len(results) >= top_k:
                break

    return results[:top_k]


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not PAGEINDEX_API_KEY:
        print("Set PAGEINDEX_API_KEY in .env first.")
    else:
        print("Checking cached PageIndex documents...")
        docs = upload_documents(wait_until_ready=False)
        print(f"Cached documents: {len(docs)}")

        print("\nTest query:")
        for result in pageindex_search("hình phạt sử dụng ma túy", top_k=3):
            print(f"[{result['score']:.3f}] {result['metadata'].get('source')}: {result['content'][:120]}...")
