$ErrorActionPreference = "Stop"

Set-Location "D:\Vin\2A202600773_Nguyen_Van_Huy_Day08_RAG_pipeline_cohort2"

.\.venv\Scripts\Activate.ps1

$env:PIP_CACHE_DIR = "D:\Vin\2A202600773_Nguyen_Van_Huy_Day08_RAG_pipeline_cohort2\.pip-cache"
$env:HF_HOME = "D:\Vin\2A202600773_Nguyen_Van_Huy_Day08_RAG_pipeline_cohort2\.hf-cache"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Write-Host "1. Crawl Task 2"
python src\task2_crawl_news.py

Write-Host "2. Convert Task 3"
python src\task3_convert_markdown.py

Write-Host "3. Build embeddings and vector index Task 4"
python src\task4_chunking_indexing.py

Write-Host "4. Smoke test retrieval modules"
python src\task5_semantic_search.py
python src\task6_lexical_search.py
python src\task7_reranking.py
python src\task8_pageindex_vectorless.py
python src\task9_retrieval_pipeline.py

Write-Host "5. Smoke test generation Task 10"
python src\task10_generation.py

Write-Host "6. Run individual grading tests"
python -m pytest tests\test_individual.py -v

Write-Host "7. Run group evaluation"
python group_project\evaluation\eval_pipeline.py
