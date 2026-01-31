# zotqa

A minimal, local-first CLI tool for querying your Zotero library with LLM assistance. Ask questions about papers you've read and notes you've taken.

## Features

- **Two-layer answers**: Every response separates your notes (your interpretations) from author claims (paper content)
- **Provenance tracking**: See exactly which chunks were used to generate each answer
- **Notes-first retrieval**: Prioritizes your notes over paper abstracts
- **LLM-agnostic**: Pluggable embedding and LLM backends (Voyage AI, OpenAI, Anthropic)
- **Single LLM call**: Optimized for one generation call per query

## Installation

```bash
# Clone and install
git clone https://github.com/yourusername/zotqa.git
cd zotqa
poetry install

# Install with embedding provider support
poetry install -E voyage  # For Voyage AI embeddings
poetry install -E openai  # For OpenAI embeddings
poetry install -E all     # For all providers
```

## Configuration

Set your API keys as environment variables:

```bash
# For embeddings (choose one)
export VOYAGE_API_KEY="your-voyage-key"
# OR
export OPENAI_API_KEY="your-openai-key"

# For LLM (Claude)
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Usage

### 1. Export your Zotero library

```bash
# Export all papers
zotqa export ./corpus

# Custom Zotero paths
zotqa export ./corpus --db ~/Zotero/zotero.sqlite --storage ~/Zotero/storage
```

This creates:
```
corpus/
├── index.json
└── papers/<key>/
    ├── metadata.json   # title, year, authors, abstract, tags
    ├── notes.md        # your notes (if any)
    └── paper.pdf       # symlink to original
```

### 2. Build the vector index

```bash
# Auto-detect embedding provider
zotqa index ./corpus

# Specify provider
zotqa index ./corpus --embedding-provider voyage
zotqa index ./corpus --embedding-provider openai

# Custom index location
zotqa index ./corpus --index-dir ~/.zotqa/my-index
```

### 3. Ask questions

```bash
# Basic query
zotqa ask "What do my notes say about protein folding?"

# Filter to specific papers
zotqa ask "What are the key findings?" --paper PAPER001 --paper PAPER002

# Control output
zotqa ask "Summarize the main themes" --max-chunks 15 --max-tokens 4096

# JSON output for programmatic use
zotqa ask "What papers discuss transformers?" --json

# Quiet mode (answer only)
zotqa ask "What is the main contribution?" -q
```

### 4. Check index status

```bash
zotqa info
```

## Answer Structure

Every answer follows a two-layer format:

```
## User-Notes Layer
[Your notes and interpretations about the topic]

## Author-Claims Layer
[What the papers claim, with citations like [PAPER001|abstract]]

## Synthesis (optional)
[When comparing multiple papers]
```

If evidence is insufficient, the tool clearly states "Insufficient evidence in corpus" rather than speculating.

## Example Session

```bash
$ zotqa ask "What did I note about attention mechanisms?"

## User-Notes Layer
Your notes highlight that attention mechanisms enable models to focus on relevant
parts of input sequences [PAPER003|notes]. You found this particularly interesting
for DNA sequence analysis applications.

## Author-Claims Layer
The paper "Transformers for Genomics" demonstrates that transformer models with
self-attention can effectively process genomic sequences [PAPER003|abstract].

---
Used chunks:
  [PAPER003|notes] (score: 0.892)
  [PAPER003|abstract] (score: 0.756)

Tokens: 1245 in / 187 out
```

## Acceptance Criteria (v1)

- [x] `index` builds searchable index and persists embeddings
- [x] `ask` returns two-layer answers with citations
- [x] Returns list of used chunks and token cost
- [x] Single LLM generation call per query
- [x] Metadata filtering via CLI flags
- [x] Conservative failure behavior (insufficient evidence / labeled inference)

## Development

```bash
# Run tests
poetry run pytest

# Format code
poetry run black src/
poetry run ruff check --fix src/

# Type checking
poetry run mypy
```

## License

MIT
