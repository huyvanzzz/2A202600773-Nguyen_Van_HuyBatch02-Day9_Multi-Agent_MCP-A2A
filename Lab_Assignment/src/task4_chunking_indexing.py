"""
Task 4 - Chunking and indexing.

Assignment notes:
- Chunking strategy: RecursiveCharacterTextSplitter.
  Ly do: corpus gom ca van ban phap luat dai va bai bao ngan, heading khong
  dong nhat. Recursive splitter uu tien tach theo doan, dong, cau roi moi tach
  theo ky tu, nen an toan hon MarkdownHeaderTextSplitter cho bo du lieu tron.
- CHUNK_SIZE=500.
  Ly do: chunk du ngan de retrieval chinh xac, nhung van du dai de giu ngu
  canh ve dieu/khoan phap luat hoac noi dung bai bao.
- CHUNK_OVERLAP=50.
  Ly do: overlap giu lai ngu canh o bien chunk, tranh mat y khi mot cau/dieu
  bi cat qua hai chunk lien tiep.
- Embedding model: sentence-transformers/all-MiniLM-L6-v2.
  Ly do: model nhe, nhanh, chay local duoc, phu hop demo/lab; embedding
  dimension la 384.
- Vector store: ChromaDB persistent local store, plus JSON backup for inspection.
  Ly do: ChromaDB la vector store local de cai va demo hon Weaviate server,
  van ho tro persistent index; JSON backup giup debug va fallback.

The JSON backup keeps the rest of the assignment easy to debug. ChromaDB is the
main vector store when the environment has been installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
INDEX_DIR = PROJECT_DIR / "data" / "index"
INDEX_PATH = INDEX_DIR / "chunks.json"
CHROMA_DIR = INDEX_DIR / "chroma"
CHROMA_COLLECTION = "drug_law_rag_chunks"

# Ghi ro theo yeu cau Task 4:
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive_character_text_splitter"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
VECTOR_STORE = "chromadb"

NOISY_PHRASES = {
    "lich phat song",
    "lịch phát sóng",
    "doi song",
    "đời sống",
    "du lich",
    "du lịch",
    "video",
    "podcast",
    "rss",
    "facebook",
    "youtube",
    "zalo",
    "advertisement",
    "quang cao",
    "quảng cáo",
}


def load_documents() -> list[dict]:
    """Read every Markdown file from data/standardized."""
    documents: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = clean_markdown(md_file.read_text(encoding="utf-8")).strip()
        if not content:
            continue
        doc_type = "legal" if md_file.parent.name == "legal" else "news"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(md_file.relative_to(PROJECT_DIR)),
                    "type": doc_type,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents with LangChain's RecursiveCharacterTextSplitter.
    Fallback splitter is provided only for missing package situations.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_text = splitter.split_text
    except Exception:
        split_text = _fallback_split_text

    chunks: list[dict] = []
    for doc in documents:
        for chunk_index, text in enumerate(split_text(doc["content"])):
            text = text.strip()
            if not is_useful_chunk(text):
                continue
            chunks.append(
                {
                    "id": _chunk_id(doc["metadata"]["path"], chunk_index, text),
                    "content": text,
                    "metadata": {**doc["metadata"], "chunk_index": chunk_index},
                }
            )
    return chunks


def _fallback_split_text(text: str) -> list[str]:
    chunks: list[str] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(text), step):
        chunks.append(text[start : start + CHUNK_SIZE])
    return chunks


def _chunk_id(path: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(f"{path}:{chunk_index}:{text[:80]}".encode("utf-8")).hexdigest()
    return digest


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def clean_markdown(text: str) -> str:
    """
    Remove crawled website navigation/noise before chunking and retrieval.
    This keeps BM25 and hybrid retrieval from ranking menu links like
    "Lich phat song", category bars, empty Markdown links, or URL-only lines.
    """
    cleaned_lines: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        if _is_noise_line(line):
            continue
        normalized = re.sub(r"\s+", " ", line.lower())
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned_lines.append(raw_line.rstrip())
    return "\n".join(cleaned_lines)


def is_useful_chunk(text: str) -> bool:
    compact = " ".join(text.split())
    if len(compact) < 40:
        return False
    if _is_noise_line(compact):
        return False
    token_count = len(tokenize(compact))
    if token_count < 8:
        return False
    return True


def _is_noise_line(line: str) -> bool:
    lower = line.lower()
    url_count = lower.count("http://") + lower.count("https://")
    markdown_link_count = len(re.findall(r"\[[^\]]*\]\([^)]+\)", line))

    if re.fullmatch(r"\[?\]?\(?https?://[^) ]+\)?", line):
        return True
    if line.startswith("[](") or line in {"[]", "[ ]"}:
        return True
    if url_count >= 2 and len(line) < 240:
        return True
    if markdown_link_count >= 3 and len(line) < 400:
        return True
    if any(phrase in lower for phrase in NOISY_PHRASES) and len(line) < 240:
        return True
    return False


def _hashed_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed with all-MiniLM-L6-v2. If the package/model is not ready yet, use a
    deterministic fallback so commands still fail gracefully during setup.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]
    except Exception as exc:
        print(f"Warning: using fallback embeddings because {exc}")
        return [_hashed_embedding(text) for text in texts]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    embeddings = embed_texts([chunk["content"] for chunk in chunks])
    return [{**chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]


def index_to_vectorstore(chunks: list[dict]) -> Path:
    """
    Index chunks into ChromaDB and also write a JSON backup.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})

        batch_size = 256
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            collection.add(
                ids=[chunk["id"] for chunk in batch],
                documents=[chunk["content"] for chunk in batch],
                metadatas=[chunk["metadata"] for chunk in batch],
                embeddings=[chunk["embedding"] for chunk in batch],
            )
        print(f"Indexed {len(chunks)} chunks into ChromaDB at {CHROMA_DIR}")
    except Exception as exc:
        print(f"Warning: ChromaDB indexing skipped: {exc}")

    return INDEX_PATH


def load_or_build_index() -> list[dict]:
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    docs = load_documents()
    chunks = embed_chunks(chunk_documents(docs))
    index_to_vectorstore(chunks)
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(size))
    norm_a = math.sqrt(sum(v * v for v in a[:size]))
    norm_b = math.sqrt(sum(v * v for v in b[:size]))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def run_pipeline() -> Path:
    docs = load_documents()
    chunks = chunk_documents(docs)
    embedded_chunks = embed_chunks(chunks)
    path = index_to_vectorstore(embedded_chunks)
    print(f"Loaded {len(docs)} cleaned docs, created {len(embedded_chunks)} useful chunks")
    return path


if __name__ == "__main__":
    run_pipeline()
