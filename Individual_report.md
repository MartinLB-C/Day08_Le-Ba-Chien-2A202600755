# Báo Cáo Cá Nhân - Task 1 Đến Task 10

## 1. Thông tin bài làm

- **Học viên:** Lê Bá Chiến
- **Mã học viên:** 2A202600755
- **Chủ đề:** RAG pipeline cho dữ liệu pháp luật Việt Nam về ma túy và tin tức nghệ sĩ liên quan đến ma túy
- **Phạm vi:** Hoàn thành các task cá nhân từ Task 1 đến Task 10 theo README

## 2. Tổng quan pipeline

Bài làm xây dựng một pipeline RAG end-to-end với luồng xử lý chính:

1. Thu thập văn bản pháp luật gốc.
2. Crawl dữ liệu tin tức.
3. Chuyển đổi dữ liệu sang Markdown.
4. Chunking, embedding và index vào FAISS.
5. Truy vấn semantic search.
6. Truy vấn lexical search bằng BM25.
7. Reranking kết quả retrieval.
8. Tích hợp PageIndex vectorless RAG.
9. Kết hợp các module thành hybrid retrieval pipeline.
10. Sinh câu trả lời có citation.

Pipeline hiện tại sử dụng FAISS làm vector store local, BM25 cho tìm kiếm từ khóa, RRF để gộp kết quả semantic và lexical, Jina Reranker hoặc fallback local để rerank, và DashScope/Qwen hoặc fallback extractive để sinh câu trả lời có nguồn.

## 3. Bảng tổng hợp trạng thái

| Task | Nội dung | File chính | Trạng thái |
|---|---|---|---|
| Task 1 | Thu thập văn bản pháp luật | `data/landing/legal/` | Hoàn thành |
| Task 2 | Crawl bài báo | `src/task2_crawl_news.py`, `data/landing/news/` | Hoàn thành |
| Task 3 | Convert sang Markdown | `src/task3_convert_markdown.py` | Hoàn thành |
| Task 4 | Chunking và indexing | `src/task4_chunking_indexing.py` | Hoàn thành |
| Task 5 | Semantic search | `src/task5_semantic_search.py` | Hoàn thành |
| Task 6 | Lexical search | `src/task6_lexical_search.py` | Hoàn thành |
| Task 7 | Reranking | `src/task7_reranking.py` | Hoàn thành |
| Task 8 | PageIndex vectorless RAG | `src/task8_pageindex_vectorless.py` | Hoàn thành có giới hạn |
| Task 9 | Retrieval pipeline | `src/task9_retrieval_pipeline.py` | Hoàn thành |
| Task 10 | Generation có citation | `src/task10_generation.py` | Hoàn thành |

## 4. Chi tiết từng task

### Task 1 - Thu thập văn bản pháp luật

Dữ liệu pháp luật gốc được lưu trong `data/landing/legal/`, gồm 4 file PDF:

- Bộ luật Hình sự sửa đổi 2017.
- Luật Phòng, chống ma túy 2021.
- Nghị định 105/2021/NĐ-CP hướng dẫn thi hành Luật Phòng, chống ma túy.
- Nghị định 28/2026/NĐ-CP về danh mục chất ma túy và tiền chất.

Kết quả đáp ứng yêu cầu README là cần tối thiểu 3 văn bản pháp luật dạng PDF/DOCX trong thư mục `data/landing/legal/`.

### Task 2 - Crawl bài báo

Dữ liệu tin tức được lưu trong `data/landing/news/`, gồm 5 file JSON. Mỗi file có metadata như:

- URL gốc.
- Tiêu đề bài báo.
- Ngày crawl.
- Nội dung bài viết.
- Nội dung Markdown nếu có.

Trong quá trình crawl, nếu Crawl4AI hoặc Playwright không chạy được do thiếu browser, code có fallback bằng `urllib` để đảm bảo vẫn lấy được dữ liệu và lưu đúng định dạng.

### Task 3 - Convert sang Markdown

File `src/task3_convert_markdown.py` được dùng để chuyển dữ liệu gốc sang Markdown.

Chức năng chính:

- Dùng MarkItDown để convert PDF/DOCX trong `data/landing/legal/`.
- Convert JSON bài báo trong `data/landing/news/` sang Markdown.
- Giữ nguyên cấu trúc thư mục con `legal/` và `news/`.

Output được lưu tại:

- `data/standardized/legal/`
- `data/standardized/news/`

Dependency đã được cấu hình là `markitdown[pdf]` để hỗ trợ đọc PDF.

### Task 4 - Chunking và indexing

File chính: `src/task4_chunking_indexing.py`.

Cấu hình đã sử dụng:

- **Chunking strategy:** `RecursiveCharacterTextSplitter`
- **Chunk size:** 1000
- **Chunk overlap:** 150
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding dimension:** 384
- **Vector store:** FAISS

Lý do chọn `RecursiveCharacterTextSplitter`: dữ liệu gồm cả văn bản pháp luật dài và bài báo, heading không luôn ổn định, nên chiến lược tách theo đoạn/dòng/câu/từ là an toàn và phù hợp.

Vector store được lưu local tại:

- `data/vectorstore/faiss.index`
- `data/vectorstore/metadata.json`

### Task 5 - Semantic search

File chính: `src/task5_semantic_search.py`.

Module này thực hiện dense retrieval trên FAISS:

- Embed query bằng cùng model `sentence-transformers/all-MiniLM-L6-v2`.
- Normalize vector để tính cosine similarity.
- Search trên FAISS index.
- Trả về danh sách kết quả có `content`, `score`, `metadata`.
- Kết quả được sắp xếp giảm dần theo score.

Hàm chính:

```python
semantic_search(query: str, top_k: int = 10) -> list[dict]
```

### Task 6 - Lexical search

File chính: `src/task6_lexical_search.py`.

Module này dùng BM25 với thư viện `rank-bm25`.

Đặc điểm:

- Corpus lấy từ `data/vectorstore/metadata.json`.
- Tokenize bằng regex Unicode để xử lý tiếng Việt tốt hơn so với `split()`.
- Trả về kết quả theo format `content`, `score`, `metadata`.
- Kết quả được sắp xếp giảm dần theo score.

BM25 giúp bổ sung khả năng match từ khóa chính xác, đặc biệt hữu ích với số điều luật, tên văn bản, thuật ngữ pháp lý và tên riêng.

Hàm chính:

```python
lexical_search(query: str, top_k: int = 10) -> list[dict]
```

### Task 7 - Reranking

File chính: `src/task7_reranking.py`.

Phương pháp chính:

- **Model:** `jina-reranker-v2-base-multilingual`
- **API:** Jina Reranker API
- **Biến môi trường:** `JINA_API_KEY`

Nếu không có API key hoặc API lỗi, code fallback sang scoring lexical local để test và demo không bị crash.

Ngoài ra file này có helper `rerank_rrf()` để gộp nhiều ranked lists bằng Reciprocal Rank Fusion.

Hàm chính:

```python
rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]
```

### Task 8 - PageIndex vectorless RAG

File chính: `src/task8_pageindex_vectorless.py`.

Module này tích hợp PageIndex SDK:

- Đọc API key từ `PAGEINDEX_API_KEY`.
- Upload PDF gốc trong `data/landing/legal/`.
- Cache `doc_id` vào `data/pageindex/documents.json`.
- Query PageIndex retrieval API.
- Chuẩn hóa output về format `content`, `score`, `metadata`.

Do PageIndex free credits bị trừ theo số trang PDF và số retrieval query, Task 8 được khóa mặc định để tránh tiêu tốn credit:

```env
PAGEINDEX_ALLOW_UPLOADS=1
PAGEINDEX_ALLOW_QUERIES=1
```

Chỉ khi bật các biến môi trường trên, module mới upload hoặc query PageIndex.

### Task 9 - Retrieval pipeline hoàn chỉnh

File chính: `src/task9_retrieval_pipeline.py`.

Pipeline hiện tại gồm các bước:

1. Chạy semantic search từ FAISS.
2. Chạy lexical search bằng BM25.
3. Gộp hai danh sách kết quả bằng Reciprocal Rank Fusion.
4. Rerank bằng Jina Reranker nếu `use_reranking=True`.
5. Trả về top-k kết quả với `source='hybrid'`.

Theo cấu hình hiện tại, PageIndex fallback không được gọi trong Task 9 để tránh tiêu tốn credit. Tham số `score_threshold` vẫn được giữ để filter kết quả.

Hàm chính:

```python
retrieve(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
    use_reranking: bool = True,
) -> list[dict]
```

### Task 10 - Generation có citation

File chính: `src/task10_generation.py`.

Module này thực hiện:

- Reorder context chunks để giảm hiện tượng lost in the middle.
- Format context kèm metadata nguồn.
- Inject context vào prompt.
- Gọi LLM qua Alibaba DashScope OpenAI-compatible API.
- Trả về câu trả lời có citation.
- Fallback sang câu trả lời extractive nếu thiếu API key hoặc API lỗi.

Cấu hình LLM:

- **Provider:** Alibaba DashScope / Model Studio
- **Model:** `qwen3.5-flash`
- **Temperature:** 0.2
- **Top-p:** 0.3
- **Top-k context:** 5

Lý do chọn `top_k=5`: đủ đa dạng bằng chứng nhưng không làm prompt quá dài.

Lý do chọn `top_p=0.3` và `temperature=0.2`: giảm độ sáng tạo, phù hợp với QA pháp lý/tin tức cần bám sát nguồn.

Hàm chính:

```python
generate_with_citation(
    query: str,
    context_chunks: list[dict] | None = None,
    top_k: int = 5,
    top_p: float = 0.3,
) -> dict
```

## 5. Cấu hình môi trường

File `.env` cần cấu hình các biến sau nếu muốn dùng đầy đủ API bên ngoài:

```env
JINA_API_KEY=your_jina_api_key
PAGEINDEX_API_KEY=your_pageindex_api_key
PAGEINDEX_ALLOW_UPLOADS=0
PAGEINDEX_ALLOW_QUERIES=0
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.5-flash
```

Nếu thiếu API key, các module vẫn có fallback local để test và demo cơ bản.

## 6. Hướng dẫn chạy

### Cài đặt dependencies

```powershell
pip install -r requirements.txt
```

### Chạy từng module

```powershell
venv\Scripts\python.exe src\task3_convert_markdown.py
venv\Scripts\python.exe src\task4_chunking_indexing.py
venv\Scripts\python.exe src\task9_retrieval_pipeline.py
venv\Scripts\python.exe src\task10_generation.py
```

### Chạy test cá nhân

```powershell
venv\Scripts\python.exe -m pytest tests/ -v
```

Hoặc chạy từng nhóm test:

```powershell
venv\Scripts\python.exe -m pytest tests/test_individual.py::TestTask1 -v
venv\Scripts\python.exe -m pytest tests/test_individual.py::TestTask5 -v
venv\Scripts\python.exe -m pytest tests/test_individual.py::TestTask10 -v
```

## 7. Trạng thái kiểm thử

Trong quá trình phát triển đã kiểm tra riêng các task chính:

- Task 3: pass.
- Task 4: pass.
- Task 5: pass.
- Task 6: pass.
- Task 7: pass.
- Task 9: pass.

Task 8 không nên chạy query PageIndex nhiều lần nếu không muốn tốn credit. Task 10 có fallback local nên vẫn có thể chạy khi thiếu cấu hình DashScope.

## 8. Hạn chế

- PageIndex fallback được khóa mặc định để tránh tốn credit.
- Reranking bằng Jina phụ thuộc vào `JINA_API_KEY`; nếu thiếu key sẽ dùng fallback lexical local.
- Generation bằng Qwen phụ thuộc vào cấu hình DashScope; nếu thiếu key sẽ dùng fallback extractive.
- Citation phụ thuộc vào metadata của chunks, nên có thể cần chuẩn hóa thêm tên nguồn và năm để hiển thị đẹp hơn.
- Dữ liệu hiện còn nhỏ, mới gồm 4 văn bản pháp luật và 5 bài báo.

## 9. Kết luận

Bài làm cá nhân đã hoàn thành đầy đủ các thành phần chính của một RAG pipeline theo README: thu thập dữ liệu, chuẩn hóa, chunking, indexing, semantic search, lexical search, reranking, PageIndex integration, hybrid retrieval và generation có citation. Pipeline có thể chạy local, có fallback khi thiếu API key, và đã được tích hợp tiếp vào app Streamlit của bài nhóm.
