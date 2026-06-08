"""
Task 2 - Crawl news articles about Vietnamese artists related to drugs.

Requirements from README:
    1. Crawl at least 5 Vietnamese news articles.
    2. Use Crawl4AI or a similar crawling library.
    3. Save output to data/landing/news/.
    4. Save one JSON file per article with metadata:
       url, title, date_crawled, content.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://vietnamnet.vn/loat-ca-si-dinh-chat-cam-ma-tuy-pha-huy-nao-bo-nguoi-tre-ra-sao-2518285.html",
    "https://tienphong.vn/bi-hai-chuyen-nghe-si-test-ma-tuy-post1847129.tpo",
    "https://baovanhoa.vn/giai-tri/ma-tuy-va-nhung-cu-nga-ngua-cua-showbiz-viet-230477.html",
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://vnexpress.net/ma-tuy-trong-loi-song-showbiz-5074606.html",
]


class ArticleHTMLParser(HTMLParser):
    """Small fallback parser for news pages when Crawl4AI is unavailable."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "iframe"}

    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._tag_stack: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self._tag_stack.append(tag)
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        text = " ".join(html.unescape(data).split())
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
            return
        if self._skip_depth:
            return
        current_tag = self._tag_stack[-1] if self._tag_stack else ""
        if current_tag in {"h1", "h2", "h3", "p", "li"} and len(text) > 20:
            self.text_parts.append(text)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    @property
    def content(self) -> str:
        return "\n\n".join(dict.fromkeys(self.text_parts)).strip()


def setup_directory() -> None:
    """Create data/landing/news/ if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def slugify_url(url: str, index: int) -> str:
    """Create a stable readable filename from an article URL."""
    stem = url.rstrip("/").split("/")[-1]
    stem = re.sub(r"\.[a-z0-9]+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", stem).strip("-").lower()
    return f"article_{index:02d}_{stem or 'news'}.json"


async def crawl_with_crawl4ai(url: str) -> dict:
    """Crawl a page using Crawl4AI."""
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    metadata = getattr(result, "metadata", {}) or {}
    content = getattr(result, "markdown", "") or getattr(result, "text", "")
    return {
        "url": url,
        "title": metadata.get("title") or "Unknown title",
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content": content.strip(),
        "content_markdown": content.strip(),
        "crawler": "crawl4ai",
    }


def crawl_with_urllib(url: str) -> dict:
    """Fallback crawler using only the Python standard library."""
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=30) as response:
        raw_html = response.read()

    page_html = raw_html.decode("utf-8", errors="replace")
    parser = ArticleHTMLParser()
    parser.feed(page_html)

    return {
        "url": url,
        "title": parser.title or "Unknown title",
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content": parser.content,
        "content_markdown": parser.content,
        "crawler": "urllib",
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl one article and return metadata plus content.

    Crawl4AI is preferred because the assignment recommends it. The urllib
    fallback keeps the script usable in a fresh environment where Crawl4AI has
    not been installed yet.
    """
    try:
        article = await crawl_with_crawl4ai(url)
    except Exception as crawl4ai_error:
        try:
            article = await asyncio.to_thread(crawl_with_urllib, url)
            article["crawl4ai_error"] = str(crawl4ai_error)
        except Exception as fallback_error:
            article = {
                "url": url,
                "title": "Crawl failed",
                "date_crawled": datetime.now(timezone.utc).isoformat(),
                "content": "",
                "content_markdown": "",
                "crawler": "failed",
                "error": str(fallback_error),
                "crawl4ai_error": str(crawl4ai_error),
            }

    required_fields = {"url", "title", "date_crawled", "content"}
    missing = required_fields - set(article)
    if missing:
        raise ValueError(f"Crawled article is missing fields: {sorted(missing)}")
    return article


async def crawl_all() -> list[Path]:
    """Crawl all configured article URLs and save each article as JSON."""
    setup_directory()
    saved_files: list[Path] = []

    for index, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        filepath = DATA_DIR / slugify_url(url, index)
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved_files.append(filepath)
        print(f"  Saved: {filepath}")

    return saved_files


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("Please fill ARTICLE_URLS before running this task.")
    else:
        asyncio.run(crawl_all())
