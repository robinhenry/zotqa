"""Export Zotero papers to RAG-friendly directory structure."""

import json
import platform
import re
import shutil
from pathlib import Path

from zotqa.extract.db import Paper


def export_papers(papers: list[Paper], output_dir: Path) -> None:
    """Export papers to directory structure with symlinked PDFs."""
    papers_dir = output_dir / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    index = []

    for paper in papers:
        paper_dir = papers_dir / paper.key
        paper_dir.mkdir(exist_ok=True)

        # Write metadata.json
        metadata = {
            "key": paper.key,
            "title": paper.title,
            "year": paper.year,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "tags": paper.tags,
        }
        (paper_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # Write notes.md
        if paper.notes:
            notes_md = _convert_notes_to_markdown(paper.notes)
            (paper_dir / "notes.md").write_text(notes_md)

        # Link or copy PDF (Windows requires copy due to symlink permissions)
        if paper.pdf_path:
            pdf_link = paper_dir / "paper.pdf"
            if pdf_link.exists() or pdf_link.is_symlink():
                pdf_link.unlink()

            # On Windows, symlinks require admin privileges, so we copy instead
            if platform.system() == "Windows":
                shutil.copy2(paper.pdf_path, pdf_link)
            else:
                pdf_link.symlink_to(paper.pdf_path)

        # Add to index
        index.append(
            {
                "key": paper.key,
                "title": paper.title,
                "year": paper.year,
                "authors": paper.authors,
                "tags": paper.tags,
                "has_notes": bool(paper.notes),
                "has_pdf": paper.pdf_path is not None,
            }
        )

    # Write index.json
    (output_dir / "index.json").write_text(json.dumps(index, indent=2))


def _convert_notes_to_markdown(notes: list[str]) -> str:
    """Convert Zotero HTML notes to markdown."""
    converted = []
    for note in notes:
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", "", note)
        # Decode common HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        # Clean up whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text.strip())
        if text:
            converted.append(text)

    return "\n\n---\n\n".join(converted)
