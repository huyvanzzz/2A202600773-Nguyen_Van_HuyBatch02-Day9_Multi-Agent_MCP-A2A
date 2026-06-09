"""Task 9 - Complete retrieval pipeline.

Assignment notes:
- Pipeline dung dung flow de bai: semantic_search + lexical_search -> merge
  bang RRF -> rerank bang MMR -> fallback PageIndex/vectorless neu score thap.
- score_threshold=0.3 la nguong bao thu: neu ket qua hybrid dau tien khong du
  tin cay, pipeline chuyen sang PageIndex/local vectorless fallback.
- top_k mac dinh la 5 de tra ve du ngu canh cho generation nhung khong lam
  prompt qua dai.
"""

try:
    from .task4_chunking_indexing import is_useful_chunk
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from task4_chunking_indexing import is_useful_chunk  # type: ignore
    from task5_semantic_search import semantic_search  # type: ignore
    from task6_lexical_search import lexical_search  # type: ignore
    from task7_reranking import rerank, rerank_rrf  # type: ignore
    from task8_pageindex_vectorless import pageindex_search  # type: ignore

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "mmr"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    1. Run semantic_search + lexical_search.
    2. Merge by RRF.
    3. Rerank with MMR for diversity.
    4. Fallback to PageIndex/local vectorless search if score is below threshold.
    """
    dense_results = _filter_results(semantic_search(query, top_k=top_k * 4))
    sparse_results = _filter_results(lexical_search(query, top_k=top_k * 4))

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 3)
    merged = [{**item, "source": "hybrid"} for item in merged]

    final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD) if use_reranking else merged[:top_k]
    final_results = [{**item, "source": "hybrid"} for item in final_results]

    if not final_results or final_results[0].get("score", 0.0) < score_threshold:
        return pageindex_search(query, top_k=top_k)
    return final_results[:top_k]


def _filter_results(results: list[dict]) -> list[dict]:
    filtered = []
    seen: set[str] = set()
    for item in results:
        content = item.get("content", "")
        if not is_useful_chunk(content):
            continue
        key = " ".join(content.split()).lower()[:200]
        if key in seen:
            continue
        seen.add(key)
        filtered.append(item)
    return filtered


if __name__ == "__main__":
    for result in retrieve("hinh phat ma tuy", top_k=3):
        print(f"[{result['score']:.3f}] [{result['source']}] {result['content'][:80]}")
