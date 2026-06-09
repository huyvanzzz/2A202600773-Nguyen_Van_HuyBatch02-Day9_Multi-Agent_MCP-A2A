"""Task 6 - Lexical search with BM25.

Assignment notes:
- Chon BM25 vi de bai mac dinh yeu cau lexical search bang BM25.
- BM25 cham diem dua tren tan suat tu khoa trong document (TF), do hiem cua tu
  trong corpus (IDF), va chuan hoa do dai document de document dai khong duoc
  uu tien qua muc.
- Khi cai du moi truong, code dung thu vien rank-bm25. Neu chua cai xong,
  fallback pure Python giu dung cong thuc BM25 de pipeline van chay duoc.
"""

from __future__ import annotations

import math
from collections import Counter

try:
    from .task4_chunking_indexing import is_useful_chunk, load_or_build_index, tokenize
except ImportError:
    from task4_chunking_indexing import is_useful_chunk, load_or_build_index, tokenize  # type: ignore


def build_bm25_index(corpus: list[dict]):
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    try:
        from rank_bm25 import BM25Okapi

        return {"backend": "rank_bm25", "bm25": BM25Okapi(tokenized_corpus), "tokenized": tokenized_corpus}
    except Exception:
        doc_freq: Counter[str] = Counter()
        for tokens in tokenized_corpus:
            doc_freq.update(set(tokens))
        avgdl = sum(len(tokens) for tokens in tokenized_corpus) / max(len(tokenized_corpus), 1)
        return {
            "backend": "pure_python",
            "tokenized": tokenized_corpus,
            "doc_freq": doc_freq,
            "avgdl": avgdl,
            "n_docs": len(corpus),
        }


def _fallback_bm25_score(query_tokens: list[str], doc_tokens: list[str], index: dict) -> float:
    k1 = 1.5
    b = 0.75
    counts = Counter(doc_tokens)
    score = 0.0
    doc_len = len(doc_tokens) or 1
    for token in query_tokens:
        df = index["doc_freq"].get(token, 0)
        if df == 0:
            continue
        idf = math.log(1 + (index["n_docs"] - df + 0.5) / (df + 0.5))
        tf = counts[token]
        denom = tf + k1 * (1 - b + b * doc_len / max(index["avgdl"], 1))
        score += idf * (tf * (k1 + 1)) / denom
    return score


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    corpus = [doc for doc in load_or_build_index() if is_useful_chunk(doc["content"])]
    index = build_bm25_index(corpus)
    query_tokens = tokenize(query)

    if index["backend"] == "rank_bm25":
        scores = index["bm25"].get_scores(query_tokens)
    else:
        scores = [_fallback_bm25_score(query_tokens, tokens, index) for tokens in index["tokenized"]]

    results = []
    for doc, score in zip(corpus, scores):
        results.append(
            {
                "content": doc["content"],
                "score": float(score),
                "metadata": doc.get("metadata", {}),
                "embedding": doc.get("embedding", []),
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return [item for item in results if item["score"] > 0][:top_k]


if __name__ == "__main__":
    for result in lexical_search("ma tuy chat cam", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}")
