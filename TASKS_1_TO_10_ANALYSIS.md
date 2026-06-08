# Phan tich cong viec Task 1 den Task 10

## Tong quan

Du an xay dung pipeline RAG cho du lieu phap luat Viet Nam ve ma tuy va tin tuc nghe si lien quan den ma tuy. Pipeline hien tai di theo luong:

1. Thu thap tai lieu goc.
2. Crawl tin tuc.
3. Chuyen doi sang Markdown.
4. Chunking, embedding va index vao FAISS.
5. Semantic search.
6. Lexical search BM25.
7. Reranking bang Jina Reranker v2.
8. PageIndex vectorless RAG, da khoa mac dinh de bao ve credit.
9. Hybrid retrieval bang RRF, khong fallback PageIndex.
10. Generation co citation bang Alibaba DashScope OpenAI-compatible API voi model `qwen3.5-flash`.

## Task 1 - Thu thap van ban phap luat

Du lieu phap luat goc duoc luu trong `data/landing/legal/`, gom 4 file PDF:

- Bo luat Hinh su sua doi 2017.
- Luat Phong, chong ma tuy 2021.
- Nghi dinh 105/2021/ND-CP.
- Nghi dinh 28/2026/ND-CP ve danh muc chat ma tuy va tien chat.

Ket qua nay dap ung yeu cau README la can toi thieu 3 file PDF/DOCX trong `data/landing/legal/`.

## Task 2 - Crawl bai bao

Du lieu tin tuc duoc luu trong `data/landing/news/`, gom 5 file JSON. Moi file co metadata nhu `url`, `title`, `date_crawled` va noi dung bai viet o `content`/`content_markdown`.

Trong qua trinh crawl, Crawl4AI/Playwright co the loi neu chua cai browser, nen pipeline co fallback crawl bang urllib. JSON da duoc format nhieu dong de de doc hon.

## Task 3 - Convert sang Markdown

File `src/task3_convert_markdown.py` da duoc hoan thien de:

- Dung MarkItDown convert PDF/DOCX trong `data/landing/legal/`.
- Convert JSON bai bao trong `data/landing/news/` sang Markdown.
- Giu cau truc thu muc con `legal/` va `news/`.

Output nam trong:

- `data/standardized/legal/`
- `data/standardized/news/`

Dependency da cap nhat thanh `markitdown[pdf]` vi MarkItDown ban mac dinh khong doc PDF neu thieu extra dependency.

## Task 4 - Chunking va Indexing

File `src/task4_chunking_indexing.py` da duoc hoan thien voi cau hinh:

- Chunking: `RecursiveCharacterTextSplitter`
- `CHUNK_SIZE = 1000`
- `CHUNK_OVERLAP = 150`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: `384`
- Vector store: `FAISS`

Ly do chon RecursiveCharacterTextSplitter: corpus gom ca van ban phap luat dai va bai bao, heading khong luon on dinh, nen cach tach theo do uu tien doan/line/cau/tu la an toan.

Vector store duoc luu local:

- `data/vectorstore/faiss.index`
- `data/vectorstore/metadata.json`

Da bo JSON index cu trong `data/index/task4_vector_index.json` vi khong con dung.

## Task 5 - Semantic Search

File `src/task5_semantic_search.py` search tren FAISS:

- Embed query bang cung model `sentence-transformers/all-MiniLM-L6-v2`.
- Normalize vector va search bang FAISS cosine similarity.
- Tra ve list dict co `content`, `score`, `metadata`.
- Ket qua sorted giam dan theo score.

Task 5 tuong thich truc tiep voi vector store FAISS cua Task 4.

## Task 6 - Lexical Search

File `src/task6_lexical_search.py` dung BM25 voi `rank-bm25`:

- Corpus lay tu `data/vectorstore/metadata.json`.
- Tokenize bang regex Unicode de xu ly tieng Viet va dau cau tot hon `split()`.
- Ham chinh: `lexical_search(query, top_k=10)`.
- Tra ve `content`, `score`, `metadata`, sorted giam dan.

BM25 bo sung kha nang match keyword chinh xac, dac biet tot voi dieu luat, so dieu, cum tu phap ly.

## Task 7 - Reranking

File `src/task7_reranking.py` dung Jina Reranker v2:

- Model: `jina-reranker-v2-base-multilingual`
- API key: `JINA_API_KEY` trong `.env`
- Ham chinh: `rerank(query, candidates, top_k=5)`

Neu khong co API key hoac API loi, code fallback sang scoring lexical local de test va demo khong crash. Khi da them API key, kiem tra thuc te cho thay `rerank_model` la `jina-reranker-v2-base-multilingual`.

File nay cung co helper `rerank_rrf()` de merge nhieu ranked lists bang Reciprocal Rank Fusion.

## Task 8 - PageIndex Vectorless RAG

File `src/task8_pageindex_vectorless.py` da tich hop PageIndex SDK:

- API key: `PAGEINDEX_API_KEY` trong `.env`
- Upload PDF goc trong `data/landing/legal/`
- Cache doc_id vao `data/pageindex/documents.json`
- Query PageIndex qua retrieval API va chuan hoa output ve `content`, `score`, `metadata`, `source='pageindex'`

Do PageIndex free credits bi tru theo so trang PDF va moi retrieval query, Task 8 hien da duoc khoa mac dinh de bao ve credit:

- `PAGEINDEX_ALLOW_UPLOADS=1` moi cho upload them.
- `PAGEINDEX_ALLOW_QUERIES=1` moi cho query retrieval.

Task 9 hien khong fallback PageIndex de tranh ton credit.

## Task 9 - Retrieval Pipeline

File `src/task9_retrieval_pipeline.py` da hoan thien pipeline hybrid:

1. Chay semantic search tu FAISS.
2. Chay lexical search BM25.
3. Merge hai ranked lists bang RRF.
4. Rerank bang Jina Reranker v2 neu `use_reranking=True`.
5. Tra ve top_k ket qua voi `source='hybrid'`.

Theo yeu cau moi, PageIndex fallback da bi bo han de khong ton credit. Pipeline van giu tham so `score_threshold` nhung chi filter ket qua, khong goi PageIndex.

## Task 10 - Generation co Citation

File `src/task10_generation.py` da duoc tao moi voi cac ham:

- `reorder_for_llm(chunks)`
- `format_context(chunks)`
- `generate_with_citation(query, context_chunks=None, top_k=5, top_p=0.3)`

LLM duoc cau hinh theo Alibaba DashScope OpenAI-compatible API:

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `DASHSCOPE_MODEL=qwen3.5-flash`

`top_k=5` duoc chon de co du bang chung nhung khong lam prompt qua dai. `top_p=0.3` va `temperature=0.2` giup cau tra loi it sang tao hon, phu hop voi QA phap ly/tin tuc can grounding.

Neu thieu cau hinh DashScope, code fallback sang cau tra loi extractive co citation tu context de test local khong bi loi.

## Cau hinh LLM Alibaba/Qwen

Trong `.env`, can them:

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.5-flash
```

Voi Singapore region, `{WorkspaceId}` phai thay bang workspace ID cua ban. Neu dung endpoint public DashScope khac, thay `DASHSCOPE_BASE_URL` theo region tu console Alibaba.

Code Task 10 dung OpenAI SDK nhu sau:

```python
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)

completion = client.chat.completions.create(
    model=os.getenv("DASHSCOPE_MODEL", "qwen3.5-flash"),
    messages=[...],
    top_p=0.3,
    temperature=0.2,
)
```

## Trang thai kiem thu

Trong qua trinh lam da chay rieng cac test:

- Task 3: pass.
- Task 4: pass.
- Task 5: pass.
- Task 6: pass.
- Task 7: pass.
- Task 9: pass.

Task 8 khong nen test retrieval tiep neu khong muon ton PageIndex credits.
