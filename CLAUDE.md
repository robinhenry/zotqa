# CLAUDE.md

This file provides guidance to Claude Code when working on this repository.

## Project Overview

**zotqa** is a Python tool for querying an LLM agent (e.g., Claude API) about notes taken while reading papers in Zotero. It enables RAG (Retrieval Augmented Generation) over your Zotero library.

## Core Requirements

### Data Extraction from Zotero

For each paper in the Zotero library, extract:

1. **Metadata**
   - Title
   - Year
   - Authors
   - Abstract
   - Tags

2. **Notes**
   - All notes attached to the paper (Zotero stores these as child items)

3. **PDF Files**
   - The actual PDF attachment(s) for each paper

### Directory Structure for RAG

Organize extracted data in a directory structure optimized for LLM RAG workflows:

```
data/
├── papers/
│   ├── <paper_id>/
│   │   ├── metadata.json      # title, year, authors, abstract, tags
│   │   ├── notes.md           # concatenated notes in markdown
│   │   └── paper.pdf          # symlink to original PDF (avoid duplication)
│   └── ...
└── index.json                 # master index of all papers
```

## Technical Context

### Zotero Data Access

Read directly from the local Zotero SQLite database (no cloud API):
- **SQLite database**: `~/Zotero/zotero.sqlite` (metadata, notes, tags)
- **Storage folder**: `~/Zotero/storage/` (PDFs and attachments, organized by 8-char item keys)

### Filtering Support

The tool should support filtering papers by:
- **Tags** - Include only papers with specific tag(s)
- **Collections** - Include only papers in specific collection(s)

### Key Dependencies to Consider

- SQLite for local database access
- PDF text extraction (e.g., `pymupdf`, `pdfplumber`)
- `anthropic` - Claude API client

## Commands

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Format code
poetry run black src/
poetry run ruff check --fix src/

# Type checking
poetry run mypy
```

## Code Style

- Python 3.11+
- Line length: 120 characters
- Use `black` for formatting, `ruff` for linting
- Type hints encouraged
