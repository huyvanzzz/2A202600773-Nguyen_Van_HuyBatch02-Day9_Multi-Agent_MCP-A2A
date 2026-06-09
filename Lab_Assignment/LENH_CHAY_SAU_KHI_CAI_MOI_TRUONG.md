# Lenh chay sau khi cai moi truong

Chay trong PowerShell:

```powershell
cd D:\Vin\2A202600773-Nguyen_Van_HuyBatch02-Day9_Multi-Agent_MCP-A2A\Lab_Assignment
.\.venv\Scripts\Activate.ps1

$env:PIP_CACHE_DIR="D:\Vin\2A202600773-Nguyen_Van_HuyBatch02-Day9_Multi-Agent_MCP-A2A\Lab_Assignment\.pip-cache"
$env:HF_HOME="D:\Vin\2A202600773-Nguyen_Van_HuyBatch02-Day9_Multi-Agent_MCP-A2A\Lab_Assignment\.hf-cache"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"

python -m pip show crawl4ai markitdown sentence-transformers rank-bm25 chromadb
python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print(m.get_sentence_embedding_dimension())"

powershell -ExecutionPolicy Bypass -File .\RUN_FULL_PIPELINE.ps1
```

Neu muon chay tung buoc:

```powershell
python src\task2_crawl_news.py
python src\task3_convert_markdown.py
python src\task4_chunking_indexing.py
python src\task5_semantic_search.py
python src\task6_lexical_search.py
python src\task7_reranking.py
python src\task8_pageindex_vectorless.py
python src\task9_retrieval_pipeline.py
python src\task10_generation.py
python -m pytest tests\test_individual.py -v
python group_project\evaluation\eval_pipeline.py
```

Task can API key:

- `PAGEINDEX_API_KEY`: de Task 8 upload/query PageIndex that.
- `OPENROUTER_API_KEY`: de Task 10 generation that qua OpenRouter.

Khong co cac key nay thi code se dung fallback local de pipeline khong bi vo.
