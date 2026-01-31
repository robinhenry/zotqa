"""Tests for the RAG pipeline."""

import json
from pathlib import Path

import pytest

from zotqa.rag import (
    Chunk,
    MockEmbedding,
    MockLLM,
    QueryEngine,
    QueryResult,
    VectorIndex,
    chunk_corpus,
    chunk_paper,
    format_answer,
)
from zotqa.rag.retrieve import Retriever


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """Create a sample corpus for testing."""
    papers_dir = tmp_path / "papers"

    # Paper 1: With notes
    paper1_dir = papers_dir / "PAPER001"
    paper1_dir.mkdir(parents=True)

    (paper1_dir / "metadata.json").write_text(
        json.dumps(
            {
                "key": "PAPER001",
                "title": "Machine Learning for Biology",
                "year": 2023,
                "authors": ["Alice Smith", "Bob Jones"],
                "abstract": "This paper explores machine learning applications in biology.",
                "tags": ["ml", "biology"],
            }
        )
    )

    (paper1_dir / "notes.md").write_text(
        "This is a great paper about ML in biology.\n\n" "Key insight: Neural networks can predict protein folding."
    )

    # Paper 2: Without notes
    paper2_dir = papers_dir / "PAPER002"
    paper2_dir.mkdir(parents=True)

    (paper2_dir / "metadata.json").write_text(
        json.dumps(
            {
                "key": "PAPER002",
                "title": "Deep Learning in Healthcare",
                "year": 2024,
                "authors": ["Carol White"],
                "abstract": "We review deep learning methods for healthcare applications.",
                "tags": ["deep-learning", "healthcare"],
            }
        )
    )

    # Paper 3: With notes
    paper3_dir = papers_dir / "PAPER003"
    paper3_dir.mkdir(parents=True)

    (paper3_dir / "metadata.json").write_text(
        json.dumps(
            {
                "key": "PAPER003",
                "title": "Transformers for Genomics",
                "year": 2024,
                "authors": ["David Lee", "Eva Chen"],
                "abstract": "Transformer models can be applied to genomic sequences.",
                "tags": ["transformers", "genomics"],
            }
        )
    )

    (paper3_dir / "notes.md").write_text(
        "Interesting application of attention mechanisms to DNA sequences.\n\n"
        "This could be relevant for my research on sequence analysis."
    )

    return tmp_path


class TestChunking:
    """Tests for chunking functionality."""

    def test_chunk_paper_with_notes(self, sample_corpus: Path) -> None:
        """Test chunking a paper with notes."""
        paper_dir = sample_corpus / "papers" / "PAPER001"
        chunks = chunk_paper(paper_dir)

        assert len(chunks) >= 3  # At least: 2 note paragraphs + abstract + metadata

        # Check that notes are priority 1
        notes_chunks = [c for c in chunks if c.section == "notes"]
        assert len(notes_chunks) >= 2
        for chunk in notes_chunks:
            assert chunk.priority == 1
            assert chunk.paper_id == "PAPER001"

        # Check that abstract is priority 3 (after notes=1, body=2)
        abstract_chunks = [c for c in chunks if c.section == "abstract"]
        assert len(abstract_chunks) == 1
        assert abstract_chunks[0].priority == 3

    def test_chunk_paper_without_notes(self, sample_corpus: Path) -> None:
        """Test chunking a paper without notes."""
        paper_dir = sample_corpus / "papers" / "PAPER002"
        chunks = chunk_paper(paper_dir)

        # Should have abstract and metadata
        assert len(chunks) >= 2

        notes_chunks = [c for c in chunks if c.section == "notes"]
        assert len(notes_chunks) == 0

    def test_chunk_corpus(self, sample_corpus: Path) -> None:
        """Test chunking an entire corpus."""
        chunks = chunk_corpus(sample_corpus)

        # Should have chunks from all 3 papers
        paper_ids = set(c.paper_id for c in chunks)
        assert paper_ids == {"PAPER001", "PAPER002", "PAPER003"}


class TestIndex:
    """Tests for index building and search."""

    def test_build_index(self, sample_corpus: Path, tmp_path: Path) -> None:
        """Test building an index."""
        index_dir = tmp_path / "index"
        embedding = MockEmbedding(dimension=64)
        index = VectorIndex(index_dir)

        stats = index.build(sample_corpus, embedding, show_progress=False)

        assert stats.num_papers == 3
        assert stats.num_chunks > 0
        assert stats.embedding_dim == 64
        assert (index_dir / "chunks.json").exists()
        assert (index_dir / "embeddings.npy").exists()
        assert (index_dir / "metadata.json").exists()

    def test_load_index(self, sample_corpus: Path, tmp_path: Path) -> None:
        """Test loading a saved index."""
        index_dir = tmp_path / "index"
        embedding = MockEmbedding(dimension=64)

        # Build index
        index1 = VectorIndex(index_dir)
        index1.build(sample_corpus, embedding, show_progress=False)

        # Load into new instance
        index2 = VectorIndex(index_dir)
        assert index2.load()
        assert len(index2.chunks) == len(index1.chunks)
        assert index2.embeddings is not None

    def test_search(self, sample_corpus: Path, tmp_path: Path) -> None:
        """Test searching the index."""
        index_dir = tmp_path / "index"
        embedding = MockEmbedding(dimension=64)
        index = VectorIndex(index_dir)
        index.build(sample_corpus, embedding, show_progress=False)

        # Search with a random query embedding
        query_vec = embedding.embed_query("machine learning biology")
        results = index.search(query_vec, top_k=5)

        assert len(results) <= 5
        assert all(isinstance(r[0], Chunk) for r in results)
        assert all(isinstance(r[1], float) for r in results)


class TestRetriever:
    """Tests for the retriever."""

    def test_retrieve(self, sample_corpus: Path, tmp_path: Path) -> None:
        """Test retrieving relevant chunks."""
        index_dir = tmp_path / "index"
        embedding = MockEmbedding(dimension=64)

        # Build index first
        index = VectorIndex(index_dir)
        index.build(sample_corpus, embedding, show_progress=False)

        # Create retriever
        retriever = Retriever(index_dir=index_dir, embedding_adapter=embedding)
        result = retriever.retrieve("machine learning", top_k=5)

        assert len(result.chunks) <= 5
        assert len(result.scores) == len(result.chunks)
        assert len(result.paper_ids) > 0


class TestQueryEngine:
    """Tests for the query engine."""

    def test_query_with_mock(self, sample_corpus: Path, tmp_path: Path) -> None:
        """Test querying with mock adapters."""
        index_dir = tmp_path / "index"
        embedding = MockEmbedding(dimension=64)

        # Build index
        index = VectorIndex(index_dir)
        index.build(sample_corpus, embedding, show_progress=False)

        # Mock LLM response with expected two-layer structure
        mock_response = """## User-Notes Layer
Your notes mention that neural networks can predict protein folding [PAPER001|notes].

## Author-Claims Layer
The paper explores machine learning applications in biology [PAPER001|abstract]."""

        llm = MockLLM(response=mock_response)
        engine = QueryEngine(
            index_dir=index_dir,
            llm_adapter=llm,
            embedding_adapter=embedding,
        )

        result = engine.query("What do my notes say about machine learning?")

        assert isinstance(result, QueryResult)
        assert "User-Notes Layer" in result.answer
        assert "Author-Claims Layer" in result.answer
        assert len(result.used_chunks) > 0
        assert result.total_tokens > 0

    def test_format_answer(self) -> None:
        """Test answer formatting."""
        from zotqa.rag.query import UsedChunk

        result = QueryResult(
            answer="## User-Notes Layer\nTest answer.",
            used_chunks=[
                UsedChunk(paper_id="PAPER001", section="notes", excerpt="Test excerpt", score=0.9),
            ],
            input_tokens=100,
            output_tokens=50,
            paper_ids=["PAPER001"],
        )

        formatted = format_answer(result)
        assert "Test answer" in formatted
        assert "PAPER001" in formatted
        assert "0.900" in formatted

    def test_result_to_dict(self) -> None:
        """Test QueryResult JSON serialization."""
        from zotqa.rag.query import UsedChunk

        result = QueryResult(
            answer="Test answer",
            used_chunks=[
                UsedChunk(paper_id="PAPER001", section="notes", excerpt="Test", score=0.9),
            ],
            input_tokens=100,
            output_tokens=50,
            paper_ids=["PAPER001"],
        )

        d = result.to_dict()
        assert d["answer"] == "Test answer"
        assert len(d["used_chunks"]) == 1
        assert d["token_cost"]["total"] == 150


class TestChunkDataclass:
    """Tests for Chunk dataclass."""

    def test_chunk_serialization(self) -> None:
        """Test chunk to/from dict."""
        chunk = Chunk(
            paper_id="TEST",
            section="notes",
            content="Test content",
            priority=1,
        )

        d = chunk.to_dict()
        restored = Chunk.from_dict(d)

        assert restored.paper_id == chunk.paper_id
        assert restored.section == chunk.section
        assert restored.content == chunk.content
        assert restored.priority == chunk.priority
