# Tom tat yeu cau bai Day 08 - RAG Pipeline v2

## Muc tieu chung

Xay dung mot RAG pipeline end-to-end cho chu de:

- Phap luat Viet Nam ve ma tuy va cac chat cam.
- Tin tuc ve nghe si Viet Nam lien quan toi ma tuy.

Pipeline can bao gom cac buoc: thu thap du lieu, chuan hoa Markdown, chunking, indexing, retrieval hybrid, fallback vectorless, reranking, generation co citation va demo/evaluation cho bai nhom.

## Cau truc thu muc chinh

- `data/landing/legal/`: luu van ban phap luat goc dang PDF/DOCX.
- `data/landing/news/`: luu bai bao crawl duoc dang JSON/HTML.
- `data/standardized/legal/`: file Markdown sau khi convert tu van ban phap luat.
- `data/standardized/news/`: file Markdown sau khi convert tu bai bao.
- `src/`: code cho cac task ca nhan tu Task 1 den Task 10.
- `tests/`: test cham diem bai ca nhan.
- `group_project/`: yeu cau va san pham bai nhom.
- `group_project/evaluation/`: golden dataset, script evaluation va bao cao ket qua.

## Yeu cau bai ca nhan

### Task 1 - Thu thap van ban phap luat

- Tai toi thieu 3 van ban phap luat ve ma tuy/cac chat cam.
- Luu file goc vao `data/landing/legal/`.
- File nen dat ten ro rang, vi du `luat-phong-chong-ma-tuy-2021.pdf`, `nghi-dinh-105-2021.pdf`.
- Goi y tai lieu: Luat Phong, chong ma tuy 2021; Nghi dinh 105/2021/ND-CP; Bo luat Hinh su 2015 sua doi 2017, chuong ve toi pham ma tuy.

### Task 2 - Crawl bai bao

- Crawl toi thieu 5 bai bao ve nghe si Viet Nam lien quan toi ma tuy.
- Luu moi bai thanh 1 file trong `data/landing/news/`, dang JSON hoac HTML.
- Moi file can co metadata: URL goc, ngay crawl, tieu de bai bao.
- Khuyen nghi dung Crawl4AI.

### Task 3 - Convert sang Markdown

- Dung MarkItDown de convert toan bo file trong `data/landing/` sang Markdown.
- Output luu vao `data/standardized/`.
- Giu cau truc thu muc con `legal/` va `news/`.
- Moi file output co ten tuong ung voi file goc, duoi `.md`.

### Task 4 - Chunking va indexing

- Chon 1 chunking strategy va 1 embedding model.
- Goi y chunking: `RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter`, `SemanticChunker`.
- Goi y embedding: `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-m3`, OpenAI `text-embedding-3-small`.
- Index toan bo Markdown vao vector store.
- Trong code can ghi ro chunking strategy, chunk size, overlap, embedding model va dimension.
- Khuyen nghi vector store: Weaviate. Co the dung ChromaDB hoac FAISS neu don gian hon.

### Task 5 - Semantic search

- Viet ham:

```python
def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    ...
```

- Tra ve list dict gom `content`, `score`, `metadata`.
- Ket qua phai duoc sap xep theo score giam dan.
- Phai dung duoc embedding model da chon o Task 4.

### Task 6 - Lexical search

- Viet ham:

```python
def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    ...
```

- Mac dinh dung BM25, co the dung `rank-bm25`.
- Tra ve list dict gom `content`, `score`, `metadata`.
- Bonus neu dung cach khac nhu TF-IDF, Elasticsearch, Weaviate BM25 va giai thich duoc co che.

### Task 7 - Reranking

- Viet ham rerank de cham lai va sap xep lai cac ket qua retrieval:

```python
def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    ...
```

- Chon 1 cach: cross-encoder reranker, Jina reranker, Qwen reranker, MMR hoac RRF.
- Output phai duoc re-score va re-order theo do lien quan voi query.

### Task 8 - PageIndex vectorless RAG

- Dang ky va dung PageIndex SDK de tao vectorless retrieval.
- Upload tai lieu len PageIndex.
- Viet ham:

```python
def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    ...
```

- Ham nay dung lam fallback khi hybrid search khong co ket qua du tot.

### Task 9 - Retrieval pipeline hoan chinh

- Ket hop semantic search, lexical search, merge/fusion, rerank va fallback PageIndex.
- Viet ham:

```python
def retrieve(query: str, top_k: int = 5, score_threshold: float = 0.3) -> list[dict]:
    ...
```

- Logic yeu cau:
  - Chay semantic search va lexical search.
  - Merge ket qua bang RRF hoac weighted fusion.
  - Rerank ket qua da merge.
  - Neu top result co score thap hon threshold thi fallback sang PageIndex.
  - Tra ve top_k ket qua.

### Task 10 - Generation co citation

- Sap xep lai chunks de tranh "lost in the middle".
- Viet ham:

```python
def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    ...

def generate_with_citation(query: str, context_chunks: list[dict]) -> str:
    ...
```

- Prompt phai yeu cau LLM tra loi co citation cho tung thong tin thuc te.
- Citation co dang `[Nguon, Nam]`.
- Neu thong tin khong nam trong context thi tra loi: `I cannot verify this information`.
- Trong code comment can giai thich top_k va top_p da chon.

## Yeu cau bai nhom

Sau khi hoan thanh bai ca nhan, nhom xay dung 1 trong 2 san pham sau.

## Lua chon 1 - RAG Chatbot

- Xay dung chatbot tra loi cau hoi ve phap luat ma tuy va tin tuc lien quan.
- Co giao dien chat bang Streamlit, Gradio hoac Chainlit.
- Cau tra loi phai co citation dua tren Task 10.
- Ho tro follow-up questions/conversation memory.
- Hien thi source documents da duoc dung.
- Stack goi y: UI -> Retrieval Task 9 -> Generation Task 10 -> Display.

## Lua chon 2 - RAG Evaluation Pipeline

- Dung 1 trong 3 framework: DeepEval, RAGAS hoac TruLens.
- Tao golden dataset toi thieu 15 cap Q&A gom question, expected_answer, expected_context.
- Chay evaluation tren toan bo golden dataset voi 4 metrics:
  - Faithfulness.
  - Answer Relevance.
  - Context Recall.
  - Context Precision.
- So sanh A/B toi thieu 2 config, vi du hybrid + rerank vs dense-only.
- Bao cao bang diem, phan tich worst performers va de xuat cai tien.

## Deliverables bai nhom

- `group_project/evaluation/golden_dataset.json`: toi thieu 15 cap Q&A.
- `group_project/evaluation/eval_pipeline.py`: script chay evaluation.
- `group_project/evaluation/results.md`: bang diem va phan tich.
- Co so sanh A/B toi thieu 2 config.
- README nhom can mo ta kien truc, phan cong va huong dan chay.
- Demo phai chay duoc local hoac deploy.
- Code can duoc push len repository chung cua nhom.

## Cham diem

### Bai ca nhan - 50 diem

| Task | Noi dung | Diem |
|---|---|---:|
| 1 | Thu thap >=3 van ban phap luat | 3 |
| 2 | Crawl >=5 bai bao | 3 |
| 3 | Convert Markdown | 4 |
| 4 | Chunking va indexing | 7 |
| 5 | Semantic search | 6 |
| 6 | Lexical search | 6 |
| 7 | Reranking | 6 |
| 8 | PageIndex query | 4 |
| 9 | Retrieval pipeline va fallback | 7 |
| 10 | Generation co citation va reorder | 4 |

### Bai nhom - 30 diem

| Tieu chi | Diem |
|---|---:|
| RAG Chatbot demo chay duoc | 8 |
| Tich hop pipeline cac thanh vien | 4 |
| Kien truc ro rang va README | 3 |
| Chatbot tra loi co citation, dung noi dung | 3 |
| Evaluation pipeline | 12 |

Trong 12 diem evaluation:

| Hang muc | Diem |
|---|---:|
| Golden dataset >=15 Q&A | 3 |
| Chay eval voi >=4 metrics | 4 |
| A/B comparison >=2 configs va phan tich | 3 |
| Bao cao worst performers | 2 |

### Bonus - 20 diem

- Demo hoac dat cau hoi khien LLM khong tra loi duoc.
- Moi cau dat duoc tinh 5 diem bonus.

## Lenh can biet

```bash
pip install -r requirements.txt
pytest tests/ -v
pytest tests/test_individual.py::TestTask1 -v
pytest tests/test_individual.py::TestTask5 -v
streamlit run app.py
chainlit run app.py
```

## Ket luan ngan

Trong repo nay, phan can lam chinh la hoan thanh 10 task ca nhan trong `src/` de tao RAG pipeline day du, sau do lam bai nhom theo 1 trong 2 huong: chatbot RAG co citation hoac evaluation pipeline. Diem so tap trung nhieu vao indexing/retrieval/reranking, pipeline tong hop, va evaluation co so sanh A/B.
