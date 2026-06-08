"""
Task 3 - Convert all files in data/landing/ to Markdown.

Requirements:
    1. Scan data/landing/ for legal documents and crawled news articles.
    2. Convert PDF/DOCX files with Microsoft MarkItDown.
    3. Convert crawled JSON articles to Markdown.
    4. Save outputs under data/standardized/ while preserving legal/news folders.
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files in data/landing/legal/ to Markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        print(f"Converting: {filepath.name}")
        result = md.convert(str(filepath))
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(result.text_content.strip() + "\n", encoding="utf-8")
        print(f"  Saved: {output_path}")


def convert_news_articles():
    """Convert crawled article JSON files in data/landing/news/ to Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        output_path = output_dir / f"{filepath.stem}.md"

        header = f"# {data.get('title') or filepath.stem}\n\n"
        header += f"**Source:** {data.get('url', 'N/A')}\n"
        header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n"
        header += f"**Original file:** {filepath.name}\n\n---\n\n"

        body = (
            data.get("content_markdown")
            or data.get("markdown")
            or data.get("content")
            or data.get("text")
            or ""
        )
        output_path.write_text(header + body.strip() + "\n", encoding="utf-8")
        print(f"  Saved: {output_path}")


def convert_all():
    """Convert all supported landing files to Markdown."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\nDone! Output at:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
