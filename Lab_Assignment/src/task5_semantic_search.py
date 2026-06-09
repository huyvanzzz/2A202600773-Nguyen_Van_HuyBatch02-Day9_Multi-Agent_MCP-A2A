"""Task 5 - Semantic search using all-MiniLM-L6-v2 and ChromaDB."""

from __future__ import annotations

try:
    from .task4_chunking_indexing import (
        CHROMA_COLLECTION,
        CHROMA_DIR,
        cosine_similarity,
        embed_texts,
        load_or_build_index,
    )
except ImportError:
    from task4_chunking_indexing import (  # type: ignore
        CHROMA_COLLECTION,
        CHROMA_DIR,
        cosine_similarity,
        embed_texts,
        load_or_build_index,
    )


def _search_chroma(query: str, top_k: int) -> list[dict] | None:
    try:
        import chromadb

        query_embedding = embed_texts([query])[0]
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(CHROMA_COLLECTION)
        raw = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        results = []
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        for content, metadata, distance in zip(documents, metadatas, distances):
            results.append(
                {
                    "content": content,
                    "score": float(1 - distance),
                    "metadata": metadata or {},
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results
    except Exception:
        return None


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    chroma_results = _search_chroma(query, top_k)
    if chroma_results is not None:
        return chroma_results[:top_k]

    query_embedding = embed_texts([query])[0]
    scored: list[dict] = []
    for chunk in load_or_build_index():
        score = cosine_similarity(query_embedding, chunk.get("embedding", []))
        scored.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
                "embedding": chunk.get("embedding", []),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hinh phat ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}")
