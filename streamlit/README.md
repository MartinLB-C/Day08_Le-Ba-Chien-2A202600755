# Streamlit RAG Interface

Giao dien chat Streamlit dung backend trong `src/`:

- Task 5: FAISS semantic search
- Task 6: BM25 lexical search
- Task 7: Jina reranking
- Task 9: Hybrid retrieval + RRF
- Task 10: Qwen generation with citation

## Chay app

Tu thu muc root cua project:

```powershell
venv\Scripts\streamlit.exe run streamlit\app.py
```

Neu port 8501 bi chiem:

```powershell
venv\Scripts\streamlit.exe run streamlit\app.py --server.port 8502
```

## Cau hinh LLM

Can co cac bien sau trong `.env` neu muon dung Qwen generation:

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.5-flash
```

Neu tat toggle `Qwen generation`, app chi hien ket qua retrieval tu corpus.
