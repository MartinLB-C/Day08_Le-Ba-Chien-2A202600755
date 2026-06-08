from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation


st.set_page_config(
    page_title="Drug Law RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        max-width: 1320px;
    }
    .source-card {
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 10px;
        background: rgba(148, 163, 184, 0.08);
    }
    .source-title {
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 4px;
    }
    .source-meta {
        color: #94a3b8;
        font-size: 12px;
        margin-bottom: 8px;
    }
    .source-body {
        font-size: 13px;
        line-height: 1.45;
    }
    .answer-box {
        border: 1px solid rgba(34, 197, 94, 0.28);
        border-radius: 8px;
        padding: 14px 16px;
        background: rgba(34, 197, 94, 0.06);
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_sources", [])
    st.session_state.setdefault("last_model", "")


def source_label(item: dict) -> str:
    metadata = item.get("metadata", {})
    return metadata.get("source") or metadata.get("path") or "Unknown source"


def compact_text(value: object, limit: int = 900) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("Chưa có source. Hãy nhập câu hỏi để truy xuất dữ liệu.")
        return

    for index, item in enumerate(sources, start=1):
        metadata = item.get("metadata", {})
        label = html.escape(source_label(item))
        score = float(item.get("score", 0.0))
        doc_type = html.escape(str(metadata.get("type", "unknown")))
        chunk_index = html.escape(str(metadata.get("chunk_index", "-")))
        preview = html.escape(compact_text(item.get("content", "")))

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">{index}. {label}</div>
                <div class="source-meta">score={score:.4f} | type={doc_type} | chunk={chunk_index}</div>
                <div class="source-body">{preview}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def answer_query(query: str, top_k: int, use_generation: bool, use_reranking: bool) -> tuple[str, list[dict], str]:
    sources = retrieve(query, top_k=top_k, use_reranking=use_reranking)

    if not use_generation:
        if not sources:
            return "Không tìm thấy context phù hợp trong corpus hiện tại.", [], "retrieval_only"
        return compact_text(sources[0].get("content", ""), limit=1200), sources, "retrieval_only"

    result = generate_with_citation(query, context_chunks=sources, top_k=top_k)
    return result.get("answer", ""), result.get("sources", sources), result.get("model", "")


def render_history() -> None:
    if not st.session_state.history:
        st.info("Chưa có hội thoại. Nhập câu hỏi ở phía trên để bắt đầu.")
        return

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["query"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            if turn.get("model"):
                st.caption(f"model: {turn['model']}")


def main() -> None:
    init_state()

    with st.sidebar:
        st.header("Retrieval")
        top_k = st.slider("Top K", min_value=2, max_value=8, value=5, step=1)
        use_reranking = st.toggle("Jina reranking", value=True)
        use_generation = st.toggle("Qwen generation", value=True)

        st.header("Backend")
        st.caption("Semantic: FAISS + all-MiniLM-L6-v2")
        st.caption("Lexical: BM25")
        st.caption("Merge: RRF")
        st.caption("Rerank: Jina Reranker v2")
        st.caption("LLM: qwen3.5-flash")
        st.caption("PageIndex fallback: disabled")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_sources = []
            st.session_state.last_model = ""
            st.rerun()

    st.title("Drug Law RAG Chat")
    st.caption("Tra cứu pháp luật ma túy và tin tức liên quan bằng backend trong thư mục src.")

    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        with st.form("query_form", clear_on_submit=False):
            query = st.text_area(
                "Câu hỏi",
                placeholder="Ví dụ: Hình phạt cho tội tàng trữ trái phép chất ma túy là gì?",
                height=92,
            )
            submitted = st.form_submit_button("Tìm kiếm và trả lời", use_container_width=True)

        if submitted:
            cleaned_query = query.strip()
            if not cleaned_query:
                st.warning("Vui lòng nhập câu hỏi.")
            else:
                with st.spinner("Đang retrieval, RRF, rerank và generation..."):
                    answer, sources, model = answer_query(
                        cleaned_query,
                        top_k=top_k,
                        use_generation=use_generation,
                        use_reranking=use_reranking,
                    )
                st.session_state.history.append(
                    {"query": cleaned_query, "answer": answer, "model": model}
                )
                st.session_state.last_sources = sources
                st.session_state.last_model = model
                st.rerun()

        st.subheader("Hội thoại")
        render_history()

    with right:
        st.subheader("Sources")
        metric_a, metric_b = st.columns(2)
        metric_a.metric("Chunks", len(st.session_state.last_sources))
        metric_b.metric("Docs", len({source_label(item) for item in st.session_state.last_sources}))

        if st.session_state.last_model:
            st.caption(f"Last model: {st.session_state.last_model}")

        render_sources(st.session_state.last_sources)


if __name__ == "__main__":
    main()
