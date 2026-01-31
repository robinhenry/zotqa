"""Chunk papers into semantic units for retrieval."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf


@dataclass
class Chunk:
    """A semantic chunk from a paper."""

    paper_id: str
    section: str  # "notes", "abstract", "metadata"
    content: str
    priority: int  # 1=notes, 2=abstract, 3=metadata (lower = higher priority)

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "section": self.section,
            "content": self.content,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            paper_id=data["paper_id"],
            section=data["section"],
            content=data["content"],
            priority=data["priority"],
        )


def chunk_paper(paper_dir: Path, include_pdf_body: bool = True) -> list[Chunk]:
    """Extract chunks from a paper directory."""
    chunks = []
    paper_id = paper_dir.name

    metadata_file = paper_dir / "metadata.json"
    notes_file = paper_dir / "notes.md"
    pdf_file = paper_dir / "paper.pdf"

    if not metadata_file.exists():
        return chunks

    with open(metadata_file) as f:
        metadata = json.load(f)

    title = metadata.get("title", "")
    authors = metadata.get("authors", [])
    year = metadata.get("year")
    abstract = metadata.get("abstract", "")
    tags = metadata.get("tags", [])

    # Chunk notes (highest priority = 1)
    if notes_file.exists():
        notes_content = notes_file.read_text().strip()
        if notes_content:
            note_chunks = _split_notes(notes_content, paper_id)
            chunks.extend(note_chunks)

    # Chunk PDF body (priority = 2)
    if include_pdf_body and pdf_file.exists():
        pdf_text = _extract_pdf_text(pdf_file.resolve())
        if pdf_text:
            body_chunks = _split_pdf_body(pdf_text, paper_id)
            chunks.extend(body_chunks)

    # Chunk abstract (priority = 3)
    if abstract:
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."
        year_str = f" ({year})" if year else ""
        header = f"{title} - {author_str}{year_str}"

        chunks.append(
            Chunk(
                paper_id=paper_id,
                section="abstract",
                content=f"{header}\n\nAbstract: {abstract}",
                priority=3,
            )
        )

    # Chunk metadata (lowest priority = 4)
    meta_parts = [f"Title: {title}"]
    if authors:
        meta_parts.append(f"Authors: {', '.join(authors)}")
    if year:
        meta_parts.append(f"Year: {year}")
    if tags:
        meta_parts.append(f"Tags: {', '.join(tags)}")

    chunks.append(
        Chunk(
            paper_id=paper_id,
            section="metadata",
            content="\n".join(meta_parts),
            priority=4,
        )
    )

    return chunks


def _split_notes(content: str, paper_id: str) -> list[Chunk]:
    """Split notes into paragraph-level chunks."""
    chunks = []

    # Split by double newlines or horizontal rules
    paragraphs = re.split(r"\n\n+|---+", content)

    for para in paragraphs:
        para = para.strip()
        if len(para) < 20:  # Skip very short fragments
            continue

        chunks.append(
            Chunk(
                paper_id=paper_id,
                section="notes",
                content=para,
                priority=1,  # Highest priority
            )
        )

    return chunks


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except Exception:
        return ""


def _split_pdf_body(content: str, paper_id: str, max_chunk_size: int = 1000) -> list[Chunk]:
    """Split PDF body text into chunks."""
    chunks = []
    if not content.strip():
        return chunks

    # Split by double newlines (paragraphs)
    paragraphs = re.split(r"\n\n+", content)

    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para = para.strip()
        # Skip very short or noisy fragments
        if len(para) < 30:
            continue
        # Skip lines that look like headers/footers (page numbers, etc.)
        if re.match(r"^\d+$", para) or len(para.split()) < 3:
            continue

        para_size = len(para)

        if current_size + para_size > max_chunk_size and current_chunk:
            # Flush current chunk
            chunks.append(
                Chunk(
                    paper_id=paper_id,
                    section="body",
                    content="\n\n".join(current_chunk),
                    priority=2,  # Between notes (1) and abstract (3)
                )
            )
            current_chunk = []
            current_size = 0

        current_chunk.append(para)
        current_size += para_size

    # Flush remaining
    if current_chunk:
        chunks.append(
            Chunk(
                paper_id=paper_id,
                section="body",
                content="\n\n".join(current_chunk),
                priority=2,
            )
        )

    return chunks


def compute_paper_hash(paper_dir: Path) -> str:
    """Compute a content hash for a paper directory.

    Hash is based on: notes content, metadata, and PDF content.
    Used for incremental indexing to detect changes.
    """
    hasher = hashlib.sha256()

    # Hash metadata
    metadata_file = paper_dir / "metadata.json"
    if metadata_file.exists():
        hasher.update(metadata_file.read_bytes())

    # Hash notes
    notes_file = paper_dir / "notes.md"
    if notes_file.exists():
        hasher.update(notes_file.read_bytes())

    # Hash PDF (just first 100KB to avoid slow hashing of large files)
    pdf_file = paper_dir / "paper.pdf"
    if pdf_file.exists():
        try:
            # Resolve symlink if needed
            real_path = pdf_file.resolve()
            with open(real_path, "rb") as f:
                hasher.update(f.read(100 * 1024))
        except Exception:
            pass

    return hasher.hexdigest()[:16]  # Truncate for storage efficiency


def chunk_corpus(corpus_dir: Path, include_pdf_body: bool = True) -> list[Chunk]:
    """Chunk all papers in a corpus directory."""
    papers_dir = corpus_dir / "papers"
    if not papers_dir.exists():
        return []

    all_chunks = []
    for paper_dir in papers_dir.iterdir():
        if paper_dir.is_dir():
            chunks = chunk_paper(paper_dir, include_pdf_body=include_pdf_body)
            all_chunks.extend(chunks)

    return all_chunks
