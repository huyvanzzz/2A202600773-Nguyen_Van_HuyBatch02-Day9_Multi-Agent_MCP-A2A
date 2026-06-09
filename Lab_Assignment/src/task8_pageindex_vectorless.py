"""Task 8 - PageIndex vectorless RAG with local fallback.

PageIndex SDK v0.2.8 exposes PageIndexClient, not PageIndex. Its upload API
accepts PDF file paths and returns doc_id values. This module therefore supports:

- PAGEINDEX_API_KEY: enables PageIndex client initialization.
- PAGEINDEX_DOC_IDS: optional comma-separated doc_id list if documents were
  uploaded manually in the PageIndex dashboard.
- PDF upload from data/landing/legal when PDF files are available.

If no API key, no PDF, or no doc_id is available, the function falls back to a
local vectorless token-overlap search so Task 9 can still run.

Note: PageIndex SDK 0.2.8 marks retrieval endpoints as deprecated, so querying
uses chat_completions(..., doc_id=..., enable_citations=True) when doc_ids exist.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    from .task4_chunking_indexing import PROJECT_DIR, load_documents, load_or_build_index, tokenize
except ImportError:
    from task4_chunking_indexing import PROJECT_DIR, load_documents, load_or_build_index, tokenize  # type: ignore

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_DOC_IDS = [doc_id.strip() for doc_id in os.getenv("PAGEINDEX_DOC_IDS", "").split(",") if doc_id.strip()]
PAGEINDEX_DOC_IDS_PATH = PROJECT_DIR / "data" / "index" / "pageindex_doc_ids.json"
LEGAL_LANDING_DIR = PROJECT_DIR / "data" / "landing" / "legal"


def _pageindex_client():
    if not PAGEINDEX_API_KEY:
        return None
    try:
        from pageindex import PageIndexClient

        return PageIndexClient(api_key=PAGEINDEX_API_KEY)
    except Exception as exc:
        print(f"Warning: PageIndex SDK unavailable or failed to initialize: {exc}")
        return None


def upload_documents() -> dict:
    """
    Upload PDF legal documents to PageIndex and persist returned doc_id values.
    If there are no PDFs yet, use existing PAGEINDEX_DOC_IDS or local fallback.
    """
    client = _pageindex_client()
    local_doc_count = len(load_documents())
    if client is None:
        return {"uploaded": local_doc_count, "backend": "local_pageindex_fallback", "api_ready": False}

    pdf_files = sorted(LEGAL_LANDING_DIR.glob("*.pdf"))
    existing_doc_ids = _load_doc_ids()
    if not pdf_files:
        return {
            "uploaded": 0,
            "backend": "pageindex",
            "api_ready": True,
            "doc_ids": len(existing_doc_ids),
            "message": "No PDF files found in data/landing/legal. Add PDFs or set PAGEINDEX_DOC_IDS.",
        }

    doc_ids = existing_doc_ids[:]
    errors = []
    for pdf_file in pdf_files:
        try:
            response = client.submit_document(str(pdf_file))
            doc_id = response.get("doc_id") or response.get("id")
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)
        except Exception as exc:
            errors.append({"file": pdf_file.name, "error": str(exc)})

    if errors and not doc_ids:
        listed_doc_ids = _list_pageindex_doc_ids(client)
        if listed_doc_ids:
            doc_ids = listed_doc_ids

    _save_doc_ids(doc_ids)
    return {
        "uploaded": max(0, len(doc_ids) - len(existing_doc_ids)),
        "backend": "pageindex",
        "api_ready": True,
        "doc_ids": len(doc_ids),
        "errors": errors,
    }


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Query PageIndex doc_ids when available; otherwise use local vectorless search.
    """
    client = _pageindex_client()
    doc_ids = _load_doc_ids()
    if client is not None and doc_ids:
        results = _pageindex_chat_search(client, doc_ids, query)
        if results:
            results.sort(key=lambda item: item["score"], reverse=True)
            return results[:top_k]

    return _local_vectorless_search(query, top_k=top_k)


def _pageindex_chat_search(client, doc_ids: list[str], query: str) -> list[dict]:
    try:
        response = client.chat_completions(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Retrieve concise evidence from the uploaded documents for this query. "
                        "Return only facts grounded in the documents.\n\n"
                        f"Query: {query}"
                    ),
                }
            ],
            doc_id=doc_ids if len(doc_ids) > 1 else doc_ids[0],
            temperature=0,
            enable_citations=True,
        )
    except Exception as exc:
        print(f"Warning: PageIndex chat query failed, using local fallback: {exc}")
        return []

    text_blocks = _extract_text_blocks(response)
    results = []
    for index, text in enumerate(text_blocks):
        clean_text = " ".join(text.split())
        if len(clean_text) < 40:
            continue
        results.append(
            {
                "content": clean_text,
                "score": 1.0 / (index + 1),
                "metadata": {"source": "pageindex_chat", "doc_ids": ",".join(doc_ids)},
                "source": "pageindex",
            }
        )
    return results


def _extract_text_blocks(value) -> list[str]:
    blocks: list[str] = []
    if isinstance(value, str):
        if len(value.strip()) >= 40:
            blocks.append(value)
    elif isinstance(value, list):
        for item in value:
            blocks.extend(_extract_text_blocks(item))
    elif isinstance(value, dict):
        preferred_keys = ["text", "content", "markdown", "answer", "snippet", "summary"]
        for key in preferred_keys:
            if key in value:
                blocks.extend(_extract_text_blocks(value[key]))
        for key, item in value.items():
            if key not in preferred_keys:
                blocks.extend(_extract_text_blocks(item))
    return blocks


def _load_doc_ids() -> list[str]:
    if PAGEINDEX_DOC_IDS:
        return PAGEINDEX_DOC_IDS
    if PAGEINDEX_DOC_IDS_PATH.exists():
        try:
            return json.loads(PAGEINDEX_DOC_IDS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_doc_ids(doc_ids: list[str]) -> None:
    if not doc_ids:
        return
    PAGEINDEX_DOC_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_DOC_IDS_PATH.write_text(json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8")


def _list_pageindex_doc_ids(client) -> list[str]:
    try:
        response = client.list_documents(limit=50)
    except Exception:
        return []

    documents = response.get("documents", []) if isinstance(response, dict) else []
    doc_ids = []
    for doc in documents:
        doc_id = doc.get("id") or doc.get("doc_id")
        if doc_id:
            doc_ids.append(doc_id)
    return doc_ids


def _local_vectorless_search(query: str, top_k: int = 5) -> list[dict]:
    query_terms = set(tokenize(query))
    results = []
    for chunk in load_or_build_index():
        terms = set(tokenize(chunk["content"]))
        overlap = len(query_terms & terms)
        score = overlap / max(len(query_terms), 1)
        results.append(
            {
                "content": chunk["content"],
                "score": float(score if score > 0 else 0.01),
                "metadata": chunk.get("metadata", {}),
                "source": "pageindex",
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


if __name__ == "__main__":
    status = upload_documents()
    print(status)
    for item in pageindex_search("ma tuy", top_k=2):
        _safe_print(f"[{item['score']:.3f}] [{item['source']}] {item['content'][:120]}")
