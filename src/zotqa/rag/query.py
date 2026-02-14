"""Query LLM with retrieved context."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from zotqa.prompts import get_system_prompt, get_user_prompt_template
from zotqa.rag.embed import EmbeddingAdapter, detect_embedding_provider, get_embedding_adapter
from zotqa.rag.index import get_default_index_dir
from zotqa.rag.llm import LLMAdapter, get_llm_adapter
from zotqa.rag.retrieve import RetrievalResult, Retriever


@dataclass
class UsedChunk:
    """A chunk that was used to generate the answer."""

    paper_id: str
    section: str
    excerpt: str
    score: float
    cite_key: str = ""

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "cite_key": self.cite_key,
            "section": self.section,
            "excerpt": self.excerpt,
            "score": round(self.score, 3),
        }


@dataclass
class QueryResult:
    """Result from a query including answer and provenance."""

    answer: str
    used_chunks: list[UsedChunk]
    input_tokens: int
    output_tokens: int
    paper_ids: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "used_chunks": [c.to_dict() for c in self.used_chunks],
            "paper_ids": self.paper_ids,
            "token_cost": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "total": self.total_tokens,
            },
        }


def _generate_cite_key(metadata: dict, used_keys: set[str]) -> str:
    """Generate a short citation key from paper metadata.

    Format: FirstAuthor (Year), e.g. "Smith (2023)"
    Handles duplicates by appending a, b, c, etc.
    """
    authors = metadata.get("authors", [])
    year = metadata.get("year", "n.d.")

    if not authors:
        first_author = "Unknown"
    else:
        # Extract last name from first author
        first_author = authors[0].split()[-1] if authors[0] else "Unknown"

    base_key = f"{first_author} ({year})"

    # Handle duplicates
    if base_key not in used_keys:
        used_keys.add(base_key)
        return base_key

    # Add suffix for duplicates
    for suffix in "abcdefghijklmnopqrstuvwxyz":
        suffixed = f"{first_author} ({year}{suffix})"
        if suffixed not in used_keys:
            used_keys.add(suffixed)
            return suffixed

    return base_key  # Fallback (shouldn't happen)


def _load_paper_metadata(corpus_path: Path, paper_id: str) -> dict:
    """Load metadata for a paper from the corpus."""
    metadata_file = corpus_path / "papers" / paper_id / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            return json.load(f)
    return {}


def _build_citation_index(paper_ids: set[str], corpus_path: Path) -> dict[str, tuple[str, dict]]:
    """Build a mapping from paper_id to (cite_key, metadata)."""
    used_keys: set[str] = set()
    citation_index: dict[str, tuple[str, dict]] = {}

    for paper_id in sorted(paper_ids):  # Sort for deterministic ordering
        metadata = _load_paper_metadata(corpus_path, paper_id)
        cite_key = _generate_cite_key(metadata, used_keys)
        citation_index[paper_id] = (cite_key, metadata)

    return citation_index


def _build_context(retrieval: RetrievalResult, citation_index: dict[str, tuple[str, dict]]) -> str:
    """Build context string from retrieved chunks."""
    parts = []

    # Add citation index header
    if citation_index:
        parts.append("=== CITATION INDEX ===")
        for paper_id, (cite_key, metadata) in citation_index.items():
            title = metadata.get("title", "Unknown title")
            parts.append(f"[{cite_key}] = {title}")
        parts.append("")

    # Group chunks by type
    notes_chunks = retrieval.notes_chunks
    author_chunks = retrieval.author_chunks

    if notes_chunks:
        parts.append("=== USER NOTES ===")
        for chunk in notes_chunks:
            cite_key = citation_index.get(chunk.paper_id, (chunk.paper_id, {}))[0]
            parts.append(f"\n[{cite_key}|notes]\n{chunk.content}")

    if author_chunks:
        parts.append("\n\n=== PAPER CONTENT ===")
        for chunk in author_chunks:
            cite_key = citation_index.get(chunk.paper_id, (chunk.paper_id, {}))[0]
            parts.append(f"\n[{cite_key}|{chunk.section}]\n{chunk.content}")

    return "\n".join(parts)


def _build_prompt(question: str, context: str) -> str:
    """Build the user prompt for the LLM."""
    template = get_user_prompt_template()
    return template.format(question=question, context=context)


class QueryEngine:
    """Engine for querying the corpus with LLM-generated answers."""

    def __init__(
        self,
        index_dir: Path | None = None,
        llm_adapter: LLMAdapter | None = None,
        embedding_adapter: EmbeddingAdapter | None = None,
    ):
        self.index_dir = index_dir or get_default_index_dir()
        self.llm_adapter = llm_adapter or get_llm_adapter("anthropic")
        if embedding_adapter:
            self.embedding_adapter = embedding_adapter
        else:
            provider = detect_embedding_provider()
            self.embedding_adapter = get_embedding_adapter(provider)
        self._retriever: Retriever | None = None

    @property
    def retriever(self) -> Retriever:
        """Lazy-load the retriever."""
        if self._retriever is None:
            self._retriever = Retriever(
                index_dir=self.index_dir,
                embedding_adapter=self.embedding_adapter,
            )
        return self._retriever

    def query(
        self,
        question: str,
        max_chunks: int = 10,
        max_tokens: int = 2048,
        paper_filter: set[str] | None = None,
    ) -> QueryResult:
        """Query the corpus and generate a two-layer answer.

        Args:
            question: The user's question
            max_chunks: Maximum number of chunks to include in context
            max_tokens: Maximum tokens for LLM response
            paper_filter: Optional set of paper IDs to filter by

        Returns:
            QueryResult with answer and provenance
        """
        # Retrieve relevant chunks
        retrieval = self.retriever.retrieve(
            query=question,
            top_k=max_chunks,
            paper_filter=paper_filter,
        )

        if not retrieval.chunks:
            return QueryResult(
                answer="No relevant content found in your library for this question.",
                used_chunks=[],
                input_tokens=0,
                output_tokens=0,
                paper_ids=[],
            )

        # Build citation index for human-readable citations
        corpus_path = Path(self.retriever.index.metadata.get("corpus_path", ""))
        citation_index = _build_citation_index(retrieval.paper_ids, corpus_path)

        # Build context and prompt
        context = _build_context(retrieval, citation_index)
        prompt = _build_prompt(question, context)

        # Generate answer
        response = self.llm_adapter.generate(
            prompt=prompt,
            system=get_system_prompt(),
            max_tokens=max_tokens,
        )

        # Build used chunks for provenance (keep paper_id for traceability)
        used_chunks = []
        for chunk, score in zip(retrieval.chunks, retrieval.scores):
            excerpt = chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
            cite_key = citation_index.get(chunk.paper_id, (chunk.paper_id, {}))[0]
            used_chunks.append(
                UsedChunk(
                    paper_id=chunk.paper_id,
                    section=chunk.section,
                    excerpt=excerpt,
                    score=score,
                    cite_key=cite_key,
                )
            )

        return QueryResult(
            answer=response.content,
            used_chunks=used_chunks,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            paper_ids=sorted(retrieval.paper_ids),
        )


def format_answer(result: QueryResult, show_chunks: bool = True, show_tokens: bool = True) -> str:
    """Format a QueryResult for human-readable output."""
    parts = [result.answer]

    if show_chunks and result.used_chunks:
        parts.append("\n---\nUsed chunks:")
        for chunk in result.used_chunks:
            cite_display = chunk.cite_key or chunk.paper_id
            parts.append(f"  [{cite_display} | {chunk.paper_id} | {chunk.section}] (score: {chunk.score:.3f})")

    if show_tokens:
        parts.append(f"\n\nTokens: {result.input_tokens} in / {result.output_tokens} out")

    return "\n".join(parts)
