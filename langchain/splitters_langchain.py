from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter,
)



def print_chunk_samples(
    docs: List[Document],
    title: str,
    max_chunks: int = 3,
    max_chars: int = 220,
) -> None:
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    print(f"Total chunks: {len(docs)}\n")

    for i, doc in enumerate(docs[:max_chunks]):
        text = doc.page_content.replace("\n", " ")
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        print(f"[Chunk {i}] len={len(doc.page_content)} chars")
        if doc.metadata:
            print(f"  metadata: {doc.metadata}")
        print(f"  preview: {text}\n")

def demo_character_splitter(base_docs: List[Document]) -> List[Document]:
    """Naive: split by character length and newline separator."""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=400,
        chunk_overlap=50,
        is_separator_regex=False,
    )
    return splitter.split_documents(base_docs)


def demo_recursive_splitter(base_docs: List[Document]) -> List[Document]:
    """Recursively try larger separators first."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(base_docs)


def demo_markdown_header_splitter(raw_markdown: str) -> List[Document]:
    """Structure-aware split: respect Markdown headers (#, ##, ###)."""
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return md_splitter.split_text(raw_markdown)


def demo_pdf_recursive_splitter(pdf_path: str) -> List[Document]:
    """
    Load a PDF and split it using RecursiveCharacterTextSplitter.

    Each page becomes a Document first (via PyPDFLoader), then we chunk
    further to make it RAG-friendly.
    """
    loader = PyPDFLoader(pdf_path)
    pdf_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
    )
    return splitter.split_documents(pdf_docs)


def demo_html_header_splitter_from_file(html_path: str) -> List[Document]:
    """
    Split an HTML file using HTMLHeaderTextSplitter.

    This respects <h1>, <h2>, <h3>, ... and stores them in metadata.
    """
    headers_to_split_on = [
        ("h1", "h1"),
        ("h2", "h2"),
        ("h3", "h3"),
    ]

    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return html_splitter.split_text_from_file(html_path)


def main() -> None:
    md_path = Path("whisper-readme.md")
    if not md_path.exists():
        print(f"Error: {md_path} not found. Place the file in the project root.")
        return

    raw_text = md_path.read_text(encoding="utf-8")

    base_doc = Document(
        page_content=raw_text,
        metadata={"source": "whisper-readme.md"},
    )
    base_docs = [base_doc]

    # Character-based chunking
    char_chunks = demo_character_splitter(base_docs)
    print_chunk_samples(char_chunks, "CharacterTextSplitter (naive)")

    # Recursive chunking
    rec_chunks = demo_recursive_splitter(base_docs)
    print_chunk_samples(rec_chunks, "RecursiveCharacterTextSplitter")

    # Markdown header-based splitting
    md_chunks = demo_markdown_header_splitter(raw_text)
    print_chunk_samples(
        md_chunks,
        "MarkdownHeaderTextSplitter (notice header metadata)",
    )
    # PDF
    PDF_PATH = "pdfs/LLMAll.pdf"
    pdf_chunks = demo_pdf_recursive_splitter(PDF_PATH)
    print_chunk_samples(pdf_chunks, f"PDF + RecursiveCharacterTextSplitter ({PDF_PATH})")

    # HTML
    HTML_PATH = "langchain-quickstart.htm"
    html_chunks = demo_html_header_splitter_from_file(HTML_PATH)
    print_chunk_samples(html_chunks, f"HTMLHeaderTextSplitter (HTML file: {HTML_PATH})")

if __name__ == "__main__":
    main()