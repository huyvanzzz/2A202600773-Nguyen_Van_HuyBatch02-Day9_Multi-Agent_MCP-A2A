"""
Task 3 - Convert toan bo file trong data/landing/ thanh Markdown.

Script nay uu tien dung MarkItDown neu co san. Neu khong, no dung
mot fallback don gian cho JSON va cac file van ban de van tao duoc
du lieu Markdown phuc vu viec hoc va testing.
"""

import json
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _get_markitdown():
    try:
        from markitdown import MarkItDown
    except Exception:
        return None
    return MarkItDown()


def _safe_read_text(filepath: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
        try:
            return filepath.read_text(encoding=encoding)
        except Exception:
            continue
    return ""


def _convert_binary_document(filepath: Path, md_converter) -> str:
    if md_converter is not None:
        try:
            result = md_converter.convert(str(filepath))
            text = getattr(result, "text_content", "") or ""
            if text.strip():
                return text
        except Exception as exc:
            print(f"Warning: MarkItDown could not convert {filepath.name}: {exc}")

    raw_text = _safe_read_text(filepath)
    if raw_text.strip():
        return (
            f"# {filepath.stem}\n\n"
            f"**Conversion note:** MarkItDown could not convert this file directly, "
            f"so the pipeline preserved readable fallback text from the original file.\n\n"
            f"---\n\n{raw_text}"
        )

    return (
        f"# Converted document: {filepath.name}\n\n"
        "MarkItDown is unavailable or could not extract readable text from this file. "
        "The original document still exists in the landing directory for downstream processing.\n"
    )


def convert_legal_docs():
    """Convert PDF/DOCX/DOC files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md_converter = _get_markitdown()

    if not legal_dir.exists():
        return

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        content = _convert_binary_document(filepath, md_converter)
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        print(f"Saved: {output_path}")


def convert_news_articles():
    """Convert JSON/HTML/MD/TXT news files trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        return

    for filepath in news_dir.iterdir():
        suffix = filepath.suffix.lower()
        if suffix == ".json":
            data = json.loads(filepath.read_text(encoding="utf-8"))
            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
            content = header + data.get("content_markdown", "")
        elif suffix in {".html", ".md", ".txt"}:
            body = _safe_read_text(filepath)
            content = f"# {filepath.stem}\n\n{body}"
        else:
            continue

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        print(f"Saved: {output_path}")


def convert_all():
    """Convert toan bo files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()


if __name__ == "__main__":
    convert_all()
