"""
Task 4 - Chunking and indexing Markdown documents.

This module reads all Markdown files from data/standardized/, chunks them, creates
embeddings with all-MiniLM-L6-v2, and indexes them into a local FAISS vector
store persisted under data/vectorstore/.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
VECTORSTORE_DIR = PROJECT_DIR / "data" / "vectorstore"
FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss.index"
METADATA_PATH = VECTORSTORE_DIR / "metadata.json"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Recursive splitting is a conservative choice for mixed legal PDFs and news
# articles because it preserves paragraphs first, then falls back to sentences
# and words when headings are sparse or inconsistent.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNKING_METHOD = "recursive"

# all-MiniLM-L6-v2 is light and 384-dimensional, suitable for a local demo and
# fast enough for the small legal/news corpus in this project.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# FAISS is a local vector store, so setup is fast and does not require Docker.
# The binary index and chunk metadata are stored in data/vectorstore/.
VECTOR_STORE = "faiss"


def _doc_type_from_path(md_file: Path) -> str:
    parts = {part.lower() for part in md_file.relative_to(STANDARDIZED_DIR).parts}
    if "legal" in parts:
        return "legal"
    if "news" in parts:
        return "news"
    return "unknown"


def load_documents() -> list[dict]:
    """
    Read all Markdown files from data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, ...}}
    """
    documents: list[dict] = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue

        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(PROJECT_DIR).as_posix()
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": relative_path,
                    "type": _doc_type_from_path(md_file),
                },
            }
        )

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents with RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
    )

    chunks: list[dict] = []
    for doc_index, doc in enumerate(documents):
        splits = splitter.split_text(doc["content"])
        for chunk_index, chunk_text in enumerate(splits):
            text = chunk_text.strip()
            if not text:
                continue

            chunks.append(
                {
                    "content": text,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                        "chunking_method": CHUNKING_METHOD,
                    },
                }
            )

    return chunks


def _embed_with_sentence_transformers(texts: list[str]) -> list[list[float]]:
    """
    Embed chunks with the selected SentenceTransformer model.

    local_files_only keeps the run reproducible in restricted environments.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return [embedding.tolist() for embedding in embeddings]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an embedding vector to each chunk.

    Returns:
        Each chunk dict with an added 'embedding': list[float]
    """
    texts = [chunk["content"] for chunk in chunks]
    embeddings = _embed_with_sentence_transformers(texts)

    embedded_chunks: list[dict] = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append({**chunk, "embedding": embedding})

    return embedded_chunks


def index_to_vectorstore(chunks: list[dict]) -> Path:
    """
    Save chunks and vectors to FAISS.

    Returns:
        Path to the local FAISS persistence directory.
    """
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = np.array([chunk["embedding"] for chunk in chunks], dtype="float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    payload = {
        "config": {
            "chunking_method": CHUNKING_METHOD,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "vector_store": VECTOR_STORE,
            "index_type": "IndexFlatIP",
            "similarity": "cosine",
        },
        "chunks": [
            {
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {}),
            }
            for chunk in chunks
        ],
    }
    METADATA_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return VECTORSTORE_DIR


def run_pipeline():
    """Run the full pipeline: load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_path = index_to_vectorstore(chunks)
    print(f"Indexed to vector store: {index_path}")


if __name__ == "__main__":
    run_pipeline()
