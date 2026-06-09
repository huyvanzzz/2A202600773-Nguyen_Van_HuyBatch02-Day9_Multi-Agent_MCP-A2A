"""Supervisor-Workers wrapper for the Day 8 RAG pipeline.

This module keeps the original Day 8 retrieval/generation pipeline intact and
adds a light Supervisor pattern with three workers:

- RetrievalWorker
- GenerationWorker
- EvaluationWorker

The supervisor coordinates them in-process so the assignment can demonstrate a
clear multi-agent style architecture without adding a second external service.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .task9_retrieval_pipeline import retrieve
    from .task10_generation import generate_with_citation
except ImportError:  # pragma: no cover
    from task9_retrieval_pipeline import retrieve  # type: ignore
    from task10_generation import generate_with_citation  # type: ignore


@dataclass
class RetrievalWorker:
    def run(self, query: str, top_k: int = 5) -> list[dict]:
        return retrieve(query, top_k=top_k)


@dataclass
class GenerationWorker:
    def run(self, query: str, top_k: int = 5) -> dict:
        return generate_with_citation(query, top_k=top_k)


@dataclass
class EvaluationWorker:
    def run(self, query: str, answer: str, sources: list[dict]) -> dict[str, Any]:
        source_count = len(sources)
        cited_sources = sum(1 for item in sources if item.get("metadata", {}).get("source"))
        answer_words = set(_tokenize(answer))
        source_words = set()
        for item in sources:
            source_words.update(_tokenize(item.get("content", "")))

        overlap = len(answer_words & source_words)
        relevance = overlap / max(len(answer_words), 1)
        citation_coverage = cited_sources / max(source_count, 1)

        return {
            "source_count": source_count,
            "citation_coverage": round(citation_coverage, 3),
            "answer_relevance": round(relevance, 3),
            "has_answer": bool(answer.strip()),
        }


class Supervisor:
    def __init__(self) -> None:
        self.retrieval_worker = RetrievalWorker()
        self.generation_worker = GenerationWorker()
        self.evaluation_worker = EvaluationWorker()

    def answer(self, query: str, top_k: int = 5) -> dict[str, Any]:
        sources = self.retrieval_worker.run(query, top_k=top_k)
        generation = self.generation_worker.run(query, top_k=top_k)
        evaluation = self.evaluation_worker.run(query, generation.get("answer", ""), sources)
        return {
            "query": query,
            "answer": generation.get("answer", ""),
            "sources": sources,
            "retrieval_source": generation.get("retrieval_source", "none"),
            "generation_backend": generation.get("generation_backend", "unknown"),
            "evaluation": evaluation,
        }


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in "".join(ch if ch.isalnum() else " " for ch in text).split() if token]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    supervisor = Supervisor()
    query = "What are the legal consequences if a company breaches a non-disclosure agreement?"
    result = supervisor.answer(query)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
