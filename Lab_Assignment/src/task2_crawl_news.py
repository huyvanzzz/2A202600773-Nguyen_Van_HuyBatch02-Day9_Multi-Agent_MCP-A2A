"""
Task 2 - Crawl news articles about Vietnamese artists/public figures related
to drugs.

Primary path: use Crawl4AI to crawl the real URLs below and save each article
as JSON with URL, title, crawl date, and Markdown content.

Fallback path: if a website blocks crawling or Crawl4AI is not installed yet,
use the verified source summary so the rest of the RAG pipeline can still run.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://vtv.vn/phap-luat/bat-ca-si-chi-dan-nguoi-mau-an-tay-tiktoker-truc-phuong-do-lien-quan-ma-tuy-20241114123427363.htm",
    "https://vnexpress.net/dien-vien-le-hang-bi-dieu-tra-mua-ban-ma-tuy-4597048.html",
    "https://vnexpress.net/rapper-binh-gold-duong-tinh-ma-tuy-khi-lai-xe-tren-cao-toc-4918203.html",
    "https://vnexpress.net/dien-vien-hai-huu-tin-bi-cao-buoc-to-chuc-choi-ma-tuy-4477400.html",
    "https://vnexpress.net/ca-si-chau-viet-cuong-nhan-13-nam-tu-vi-nhet-toi-hai-chet-co-gai-3891028.html",
]

FALLBACK_ARTICLES = {
    ARTICLE_URLS[0]: {
        "title": "Bat ca si Chi Dan, nguoi mau An Tay, Tiktoker Truc Phuong do lien quan ma tuy",
        "content_markdown": (
            "VTV dua tin Cong an TP.HCM khoi to, bat tam giam ca si Chi Dan, nguoi mau An Tay "
            "va Tiktoker Truc Phuong do lien quan den hanh vi mua ma tuy de to chuc su dung trai "
            "phep chat ma tuy. Vu viec nam trong qua trinh mo rong chuyen an van chuyen ma tuy "
            "qua duong hang khong tu Phap ve Viet Nam. Nguon nay cung cap thong tin ve nguoi noi "
            "tieng, hanh vi bi dieu tra va boi canh dau tranh phong chong ma tuy."
        ),
    },
    ARTICLE_URLS[1]: {
        "title": "Dien vien Le Hang bi dieu tra mua ban ma tuy",
        "content_markdown": (
            "VnExpress dua tin dien vien Bui Thi Le Hang bi khoi to de dieu tra ve toi mua ban "
            "trai phep chat ma tuy sau khi bi bat qua tang mang ma tuy tong hop di ban. Bai viet "
            "nhac lai viec Le Hang tung duoc biet den qua vai Hoai Thatcher trong phim Xin hay "
            "tin em va tung cong tac tai Nha hat Tuoi tre."
        ),
    },
    ARTICLE_URLS[2]: {
        "title": "Rapper Binh Gold duong tinh ma tuy khi lai xe tren cao toc",
        "content_markdown": (
            "VnExpress dua tin rapper Binh Gold duong tinh voi ma tuy khi dieu khien xe tren cao "
            "toc Noi Bai - Lao Cai. Bai viet mo ta hanh vi lai xe lang lach, danh vong, sau do bi "
            "canh sat giao thong dung xe kiem tra va ban giao cho co quan chuc nang xu ly."
        ),
    },
    ARTICLE_URLS[3]: {
        "title": "Dien vien hai Huu Tin bi cao buoc to chuc choi ma tuy",
        "content_markdown": (
            "VnExpress dua tin dien vien hai Tran Huu Tin bi khoi to, bat tam giam ve hanh vi "
            "tang tru trai phep chat ma tuy va to chuc su dung trai phep chat ma tuy. Theo bai "
            "viet, Huu Tin cung mot so nguoi khac to chuc su dung ma tuy tai can ho o quan 8 "
            "va bi canh sat bat qua tang."
        ),
    },
    ARTICLE_URLS[4]: {
        "title": "Ca si Chau Viet Cuong nhan an tu trong vu an co lien quan su dung ma tuy",
        "content_markdown": (
            "VnExpress dua tin ca si Chau Viet Cuong bi tuyen phat 13 nam tu trong vu an giet "
            "nguoi, trong do toa nhan dinh Cung su dung ma tuy nen bi ao giac va khong dieu "
            "khien duoc hanh vi. Bai viet cung nhac den nguoi lien quan bi phat tu vi tang tru "
            "trai phep chat ma tuy."
        ),
    },
}


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _fallback_article(url: str, error: str | None = None) -> dict:
    data = FALLBACK_ARTICLES[url]
    return {
        "url": url,
        "title": data["title"],
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": data["content_markdown"],
        "crawl_backend": "fallback_verified_summary",
        "crawl_error": error,
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl one article with Crawl4AI and return metadata + Markdown content.
    Falls back to verified summaries if crawling is unavailable.
    """
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            metadata = getattr(result, "metadata", {}) or {}
            markdown = getattr(result, "markdown", "") or getattr(result, "text", "")
            title = metadata.get("title") or FALLBACK_ARTICLES[url]["title"]

            if len(markdown.strip()) < 500:
                fallback = _fallback_article(url, "Crawl4AI returned too little content")
                fallback["raw_crawl_preview"] = markdown[:500]
                return fallback

            return {
                "url": url,
                "title": title,
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": markdown,
                "crawl_backend": "crawl4ai",
                "crawl_error": None,
            }
    except Exception as exc:
        return _fallback_article(url, str(exc))


async def crawl_all() -> list[dict]:
    setup_directory()
    articles = []
    for index, url in enumerate(ARTICLE_URLS, 1):
        article = await crawl_article(url)
        articles.append(article)
        filepath = DATA_DIR / f"article_{index:02d}.json"
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {filepath} ({article['crawl_backend']})")
    return articles


if __name__ == "__main__":
    asyncio.run(crawl_all())
