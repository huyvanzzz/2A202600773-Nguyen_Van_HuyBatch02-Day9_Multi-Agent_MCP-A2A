"""Offline A/B evaluation for the Day 8 group assignment.

This pipeline avoids hard dependency on strong external models. It compares two
retrieval configurations against the golden dataset and writes results.md.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation
from src.task6_lexical_search import lexical_search
from src.task5_semantic_search import semantic_search

DATASET_PATH = Path(__file__).with_name("golden_dataset.json")
RESULTS_PATH = Path(__file__).with_name("results.md")


@dataclass
class EvaluationResult:
    config_name: str
    average_metrics: dict[str, float]
    rows: list[dict[str, Any]]


def load_dataset() -> list[dict[str, Any]]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def run_config_a(question: str) -> dict[str, Any]:
    sources = retrieve(question, top_k=5)
    generation = generate_with_citation(question, top_k=5)
    return {
        "sources": sources,
        "answer": generation["answer"],
        "backend": generation.get("generation_backend", "unknown"),
    }


def run_config_b(question: str) -> dict[str, Any]:
    dense = semantic_search(question, top_k=5)
    sparse = lexical_search(question, top_k=5)
    merged = _merge_results(dense, sparse, top_k=5)
    answer = _fallback_answer(question, merged)
    return {"sources": merged, "answer": answer, "backend": "hybrid_no_rerank"}


def _merge_results(dense: list[dict], sparse: list[dict], top_k: int = 5) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list in (dense, sparse):
        for rank, item in enumerate(ranked_list, 1):
            key = _dedupe_key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (50 + rank)
            items[key] = item
    merged = []
    for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True):
        merged.append({**items[key], "score": float(score)})
    return merged[:top_k]


def _dedupe_key(item: dict) -> str:
    metadata = item.get("metadata", {})
    source = metadata.get("source", "")
    chunk_index = metadata.get("chunk_index", "")
    if source and chunk_index != "":
        return f"{source}:{chunk_index}"
    return item.get("content", "")


def _fallback_answer(question: str, sources: list[dict]) -> str:
    if not sources:
        return "I cannot verify this information."
    bullets = []
    for item in sources[:3]:
        source = item.get("metadata", {}).get("source", "unknown")
        content = " ".join(item.get("content", "").split())[:220]
        bullets.append(f"- {content} [{source}]")
    return f"Evidence for: {question}\n" + "\n".join(bullets)


def evaluate_row(question: str, gold_answer: str, keywords: list[str], runner) -> dict[str, Any]:
    result = runner(question)
    sources = result["sources"]
    answer = result["answer"]
    source_text = " ".join(item.get("content", "") for item in sources).lower()
    answer_text = answer.lower()
    keyword_hits = sum(1 for kw in keywords if kw.lower() in source_text or kw.lower() in answer_text)

    citation_coverage = sum(1 for item in sources if item.get("metadata", {}).get("source")) / max(len(sources), 1)
    answer_relevance = keyword_hits / max(len(keywords), 1)
    context_recall = keyword_hits / max(len(keywords), 1)
    context_precision = keyword_hits / max(len(sources), 1)

    return {
        "question": question,
        "gold_answer": gold_answer,
        "answer": answer,
        "backend": result["backend"],
        "citation_coverage": round(citation_coverage, 3),
        "answer_relevance": round(answer_relevance, 3),
        "context_recall": round(context_recall, 3),
        "context_precision": round(context_precision, 3),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ["citation_coverage", "answer_relevance", "context_recall", "context_precision"]
    return {key: round(sum(row[key] for row in rows) / max(len(rows), 1), 3) for key in keys}


def format_results_md(result_a: EvaluationResult, result_b: EvaluationResult) -> str:
    score_a = sum(result_a.average_metrics.values()) / max(len(result_a.average_metrics), 1)
    score_b = sum(result_b.average_metrics.values()) / max(len(result_b.average_metrics), 1)
    winner = "Config A" if score_a >= score_b else "Config B"
    lines = [
        "# Group Evaluation Results",
        "",
        "## Config A",
        "",
        f"- Backend: `{result_a.rows[0]['backend'] if result_a.rows else 'n/a'}`",
        "",
        "| Metric | Score |",
        "|---|---:|",
    ]
    for metric, value in result_a.average_metrics.items():
        lines.append(f"| {metric} | {value:.3f} |")
    lines += ["", "## Config B", "", f"- Backend: `{result_b.rows[0]['backend'] if result_b.rows else 'n/a'}`", "", "| Metric | Score |", "|---|---:|"]
    for metric, value in result_b.average_metrics.items():
        lines.append(f"| {metric} | {value:.3f} |")
    lines += [
        "",
        "## Notes",
        "",
        "- Config A uses the full retrieval pipeline with reranking.",
        "- Config B uses dense + lexical merge without reranking.",
        f"- Winner on this run: `{winner}`.",
    ]
    return "\n".join(lines)


def main() -> None:
    dataset = load_dataset()
    rows_a = [evaluate_row(item["question"], item["answer"], item.get("keywords", []), run_config_a) for item in dataset]
    rows_b = [evaluate_row(item["question"], item["answer"], item.get("keywords", []), run_config_b) for item in dataset]
    result_a = EvaluationResult("config_a", summarize(rows_a), rows_a)
    result_b = EvaluationResult("config_b", summarize(rows_b), rows_b)

    RESULTS_PATH.write_text(format_results_md(result_a, result_b), encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    print("Config A:", result_a.average_metrics)
    print("Config B:", result_b.average_metrics)


if __name__ == "__main__":
    main()
