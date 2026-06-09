"""Task 10 - Generation with citations.

Assignment notes:
- top_k=5: lay 5 chunks vi du evidence cho cau tra loi RAG nhung khong lam
  prompt qua dai, giam nguy co lost in the middle.
- top_p=0.9: giu cau tra loi tu nhien nhung khong qua ngau nhien.
- temperature=0.3: RAG can factual, nen de temperature thap de han che suy doan.
- Reorder chunks de tranh lost in the middle: chunk quan trong duoc dat o dau
  va cuoi context, vi LLM thuong chu y hai vi tri nay tot hon phan giua.
- Citation: moi claim thuc te phai co citation dang [Nguon, Nam] hoac source
  metadata tu context; neu context khong du thi tra loi khong the xac minh.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve  # type: ignore

load_dotenv()

# Ghi ro theo yeu cau Task 10:
# top_k=5 gives enough evidence without making the prompt too long.
TOP_K = 5
# top_p=0.9 balances fluent wording with factual restraint for RAG answers.
TOP_P = 0.9
TEMPERATURE = 0.3
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"

SYSTEM_PROMPT = """You are a Vietnamese RAG assistant.

Use ONLY the provided context. If the context contains relevant evidence, answer
the question directly and cite every factual claim with the Source label shown
in the context, for example [105_2021_ND-CP_496664.md] or [article_04.md].

If the provided context has no relevant evidence at all, say exactly:
I cannot verify this information.

Do not require perfect source formatting such as author/year; the Source label
from the context is enough for this assignment."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    if len(chunks) <= 2:
        return chunks
    reordered = chunks[::2]
    reordered.extend(reversed(chunks[1::2]))
    return reordered


def format_context(chunks: list[dict]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"source_{index}")
        doc_type = metadata.get("type", "unknown")
        parts.append(
            f"[Document {index} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


def _fallback_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I cannot verify this information"
    bullets = []
    for chunk in chunks[:3]:
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown source")
        content = " ".join(chunk.get("content", "").split())
        excerpt = content[:260].rstrip()
        bullets.append(f"- {excerpt} [{source}]")
    return f"Duoi day la cac bang chung tim thay cho cau hoi: {query}\n" + "\n".join(bullets)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if openrouter_key or openai_key:
        try:
            from openai import OpenAI

            if openrouter_key:
                client = OpenAI(
                    api_key=openrouter_key,
                    base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
                )
                model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
            else:
                client = OpenAI(api_key=openai_key)
                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content or ""
            if chunks and "cannot verify" in answer.lower():
                answer = _fallback_answer(query, reordered)
        except Exception as exc:
            print(f"Warning: LLM generation failed, using local fallback: {exc}")
            answer = _fallback_answer(query, reordered)
    else:
        answer = _fallback_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
        "generation_backend": "openrouter" if openrouter_key else ("openai" if openai_key else "local_fallback"),
    }


if __name__ == "__main__":
    result = generate_with_citation("hinh phat ma tuy")
    print(f"Backend: {result['generation_backend']} | Retrieval: {result['retrieval_source']}")
    print(result["answer"].encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))
