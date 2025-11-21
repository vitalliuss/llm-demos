from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader
from bs4 import BeautifulSoup

from semantic_kernel.text import text_chunker as tc
# https://learn.microsoft.com/python/api/semantic-kernel/semantic_kernel.text.text_chunker


def print_chunk_samples(
    chunks: List[str],
    title: str,
    max_chunks: int = 3,
    max_chars: int = 220,
) -> None:
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    print(f"Total chunks: {len(chunks)}\n")

    for i, text in enumerate(chunks[:max_chunks]):
        one_line = text.replace("\n", " ")
        if len(one_line) > max_chars:
            one_line = one_line[: max_chars - 3] + "..."
        print(f"[Chunk {i}] len={len(text)} chars")
        print(f"  preview: {one_line}\n")


# ------------------------------
# Markdown + plain text splitters
# ------------------------------

def demo_plaintext_chunker(raw_text: str) -> List[str]:
    """
    Use SK's plaintext chunker:
    - First split into lines (by newlines, then punctuation)
    - Then group lines into paragraphs/chunks
    """
    lines = tc.split_plaintext_lines(
        text=raw_text,
        max_token_per_line=200,
    )
    paragraphs = tc.split_plaintext_paragraph(
        text=lines,
        max_tokens=400,
    )
    return paragraphs


def demo_markdown_chunker(raw_markdown: str) -> List[str]:
    """
    Use SK's markdown chunker:
    - First split into markdown-aware lines
    - Then group lines into paragraphs/chunks
    """
    lines = tc.split_markdown_lines(
        text=raw_markdown,
        max_token_per_line=200,
    )
    paragraphs = tc.split_markdown_paragraph(
        text=lines,
        max_tokens=400,
    )
    return paragraphs


def load_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n\n".join(pages)


def demo_pdf_chunker(pdf_path: str) -> List[str]:
    pdf_text = load_pdf_text(pdf_path)
    return demo_plaintext_chunker(pdf_text)


def load_html_text(html_path: str) -> str:
    html = Path(html_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    return text


def demo_html_chunker(html_path: str) -> List[str]:
    html_text = load_html_text(html_path)
    return demo_plaintext_chunker(html_text)


def main() -> None:
    md_path = Path("whisper-readme.md")
    if not md_path.exists():
        print(f"Error: {md_path} not found. Place the file in the project root.")
        return

    raw_text = md_path.read_text(encoding="utf-8")

    # Plain text-style chunking
    plain_chunks = demo_plaintext_chunker(raw_text)
    print_chunk_samples(plain_chunks, "Semantic Kernel: split_plaintext_* on Markdown file")

    # Markdown-aware chunking
    md_chunks = demo_markdown_chunker(raw_text)
    print_chunk_samples(md_chunks, "Semantic Kernel: split_markdown_* on Markdown file")

    # PDF
    PDF_PATH = "pdfs/LLMAll.pdf"
    pdf_chunks = demo_pdf_chunker(PDF_PATH)
    print_chunk_samples(pdf_chunks, f"Semantic Kernel: plaintext chunking on PDF ({PDF_PATH})")

    # HTML
    HTML_PATH = "langchain-quickstart.htm"
    html_chunks = demo_html_chunker(HTML_PATH)
    print_chunk_samples(html_chunks, f"Semantic Kernel: plaintext chunking on HTML ({HTML_PATH})")


if __name__ == "__main__":
    main()