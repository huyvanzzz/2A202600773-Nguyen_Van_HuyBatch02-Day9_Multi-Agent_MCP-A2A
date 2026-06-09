"""Task 7 - Reranking with RRF and MMR.

Assignment notes:
- Chon RRF de merge ket qua tu semantic search va lexical search.
  RRF phu hop khi hai retriever co thang diem khac nhau, vi no dua vao rank
  thay vi so sanh truc tiep score dense/BM25.
- Chon MMR de rerank sau khi merge.
  MMR vua giu do lien quan voi query, vua giam trung lap giua cac chunks. Dieu
  nay huu ich cho RAG vi LLM can bang chung da dang, khong phai nhieu chunk lap
  lai cung mot noi dung.
- Khong dung cross-encoder vi de bai cho phep tu implement RRF/MMR va cach nay
  khong can API key hay model reranker lon.
"""

from __future__ import annotations

try:
    from .task4_chunking_indexing import cosine_similarity, embed_texts
except ImportError:
    from task4_chunking_indexing import cosine_similarity, embed_texts  # type: ignore


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion.

    RRF(d) = sum(1 / (k + rank_r(d))) across rankers.
    This is used to merge semantic and lexical retrieval results.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = _dedupe_key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items[key] = item

    merged = []
    for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True):
        merged.append({**items[key], "score": float(score), "rerank_method": "rrf"})
    return merged[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance.

    MMR balances relevance to the query with diversity against already selected
    chunks. lambda_param=0.7 favors relevance while still reducing duplicates.
    """
    if not candidates:
        return []

    prepared = []
    missing_embedding_texts = [item["content"] for item in candidates if not item.get("embedding")]
    generated_embeddings = iter(embed_texts(missing_embedding_texts)) if missing_embedding_texts else iter(())

    for item in candidates:
        embedding = item.get("embedding") or next(generated_embeddings)
        relevance = cosine_similarity(query_embedding, embedding)
        prepared.append({**item, "embedding": embedding, "_query_relevance": relevance})

    selected: list[dict] = []
    remaining = prepared[:]

    while remaining and len(selected) < top_k:
        best_index = 0
        best_score = float("-inf")
        for index, candidate in enumerate(remaining):
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    cosine_similarity(candidate["embedding"], chosen["embedding"]) for chosen in selected
                )
            mmr_score = lambda_param * candidate["_query_relevance"] - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_index = index
                best_score = mmr_score

        chosen = remaining.pop(best_index)
        cleaned = {key: value for key, value in chosen.items() if key != "_query_relevance"}
        cleaned["score"] = float(best_score)
        cleaned["rerank_method"] = "mmr"
        selected.append(cleaned)

    return selected


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "mmr",
) -> list[dict]:
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "mmr":
        query_embedding = embed_texts([query])[0]
        return rerank_mmr(query_embedding, candidates, top_k=top_k)
    raise ValueError(f"Unsupported rerank method without API key: {method}")


def _dedupe_key(item: dict) -> str:
    metadata = item.get("metadata", {})
    source = metadata.get("source", "")
    chunk_index = metadata.get("chunk_index", "")
    if source != "" and chunk_index != "":
        return f"{source}:{chunk_index}"
    return item.get("content", "")


if __name__ == "__main__":
    docs = [
        {"content": "Toi tang tru ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Tin tuc nghe si lien quan ma tuy", "score": 0.6, "metadata": {}},
    ]
    print(rerank("hinh phat ma tuy", docs))
