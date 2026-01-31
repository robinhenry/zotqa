"""Retrieve relevant papers from vector index."""

from dataclasses import dataclass
from pathlib import Path

from zotqa.rag.chunk import Chunk
from zotqa.rag.embed import EmbeddingAdapter, detect_embedding_provider, get_embedding_adapter
from zotqa.rag.index import VectorIndex, get_default_index_dir


@dataclass
class RetrievalResult:
    """Result from retrieval containing ranked chunks."""

    chunks: list[Chunk]
    scores: list[float]
    paper_ids: set[str]

    @property
    def notes_chunks(self) -> list[Chunk]:
        """Get only chunks from notes."""
        return [c for c in self.chunks if c.section == "notes"]

    @property
    def author_chunks(self) -> list[Chunk]:
        """Get only chunks from author content (body, abstract, metadata)."""
        return [c for c in self.chunks if c.section in ("body", "abstract", "metadata")]


class Retriever:
    """Retrieves relevant chunks for a query."""

    def __init__(
        self,
        index_dir: Path | None = None,
        embedding_adapter: EmbeddingAdapter | None = None,
    ):
        self.index_dir = index_dir or get_default_index_dir()
        self.index = VectorIndex(self.index_dir)
        self.embedding_adapter = embedding_adapter

        if not self.index.load():
            raise ValueError(f"No index found at {self.index_dir}. Run 'zotqa index' first.")

        if self.embedding_adapter is None:
            provider = detect_embedding_provider()
            self.embedding_adapter = get_embedding_adapter(provider)

    def retrieve(
        self,
        query: str,
        top_k: int = 12,
        paper_filter: set[str] | None = None,
        notes_boost: float = 1.5,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        Args:
            query: The search query
            top_k: Maximum number of chunks to return
            paper_filter: Optional set of paper IDs to filter by
            notes_boost: Score multiplier for notes chunks (prioritizes user notes)

        Returns:
            RetrievalResult with ranked chunks
        """
        # Embed the query
        query_embedding = self.embedding_adapter.embed_query(query)

        # Search with extra candidates for reranking
        candidates = self.index.search(
            query_embedding,
            top_k=top_k * 2,  # Fetch more for reranking
            paper_filter=paper_filter,
        )

        if not candidates:
            return RetrievalResult(chunks=[], scores=[], paper_ids=set())

        # Apply notes boost (prioritize user notes)
        reranked = []
        for chunk, score in candidates:
            if chunk.section == "notes":
                score *= notes_boost
            reranked.append((chunk, score))

        # Sort by boosted score and take top_k
        reranked.sort(key=lambda x: x[1], reverse=True)
        reranked = reranked[:top_k]

        # Deduplicate by content similarity (keep first occurrence)
        seen_content = set()
        deduped = []
        for chunk, score in reranked:
            content_key = chunk.content[:100]  # Use prefix as key
            if content_key not in seen_content:
                seen_content.add(content_key)
                deduped.append((chunk, score))

        chunks = [c for c, _ in deduped]
        scores = [s for _, s in deduped]
        paper_ids = set(c.paper_id for c in chunks)

        return RetrievalResult(chunks=chunks, scores=scores, paper_ids=paper_ids)
