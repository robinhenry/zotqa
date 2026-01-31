"""Build and manage vector index for papers."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from zotqa.rag.chunk import Chunk, chunk_paper, compute_paper_hash
from zotqa.rag.embed import EmbeddingAdapter, get_embedding_adapter


@dataclass
class IndexStats:
    """Statistics about the index."""

    num_papers: int
    num_chunks: int
    embedding_dim: int
    indexed_at: str
    corpus_path: str


class VectorIndex:
    """Simple vector index with cosine similarity search."""

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.metadata: dict = {}

    def build(
        self,
        corpus_dir: Path,
        embedding_adapter: EmbeddingAdapter | None = None,
        show_progress: bool = True,
        force_rebuild: bool = False,
        include_pdf_body: bool = True,
    ) -> IndexStats:
        """Build index from a corpus directory.

        Supports incremental indexing: only new/changed papers are re-embedded.
        """
        if embedding_adapter is None:
            embedding_adapter = get_embedding_adapter("voyage")

        papers_dir = corpus_dir / "papers"
        if not papers_dir.exists():
            raise ValueError(f"No papers directory found in corpus: {corpus_dir}")

        # Compute current hashes for all papers
        paper_dirs = [d for d in papers_dir.iterdir() if d.is_dir()]
        current_hashes: dict[str, str] = {}
        hash_iter = paper_dirs
        if show_progress:
            hash_iter = tqdm(paper_dirs, desc="Scanning papers", unit="paper")
        for paper_dir in hash_iter:
            current_hashes[paper_dir.name] = compute_paper_hash(paper_dir)

        # Load existing index if available (for incremental update)
        existing_hashes: dict[str, str] = {}
        if not force_rebuild and self.load():
            existing_hashes = self.metadata.get("paper_hashes", {})

        # Determine what changed
        new_papers = set(current_hashes.keys()) - set(existing_hashes.keys())
        deleted_papers = set(existing_hashes.keys()) - set(current_hashes.keys())
        changed_papers = {
            pid for pid in current_hashes
            if pid in existing_hashes and current_hashes[pid] != existing_hashes[pid]
        }
        papers_to_index = new_papers | changed_papers
        unchanged_papers = set(current_hashes.keys()) - papers_to_index

        if show_progress:
            print(f"Papers: {len(new_papers)} new, {len(changed_papers)} changed, "
                  f"{len(deleted_papers)} deleted, {len(unchanged_papers)} unchanged")

        # If nothing changed and index exists, return early
        if not papers_to_index and not deleted_papers and self.chunks:
            if show_progress:
                print("Index is up to date.")
            return IndexStats(
                num_papers=self.metadata["num_papers"],
                num_chunks=self.metadata["num_chunks"],
                embedding_dim=self.metadata["embedding_dim"],
                indexed_at=self.metadata["indexed_at"],
                corpus_path=self.metadata["corpus_path"],
            )

        # Keep chunks/embeddings for unchanged papers
        kept_chunks: list[Chunk] = []
        kept_embeddings: list[np.ndarray] = []
        if self.chunks and self.embeddings is not None:
            for i, chunk in enumerate(self.chunks):
                if chunk.paper_id in unchanged_papers:
                    kept_chunks.append(chunk)
                    kept_embeddings.append(self.embeddings[i])

        # Chunk new/changed papers
        new_chunks: list[Chunk] = []
        chunk_iter = papers_to_index
        if show_progress and papers_to_index:
            chunk_iter = tqdm(papers_to_index, desc="Extracting text", unit="paper")
        for paper_id in chunk_iter:
            paper_dir = papers_dir / paper_id
            chunks = chunk_paper(paper_dir, include_pdf_body=include_pdf_body)
            new_chunks.extend(chunks)

        # Embed new chunks
        new_embeddings: np.ndarray | None = None
        if new_chunks:
            texts = [c.content for c in new_chunks]
            new_embeddings = embedding_adapter.embed(texts)

        # Merge kept and new
        self.chunks = kept_chunks + new_chunks
        if kept_embeddings and new_embeddings is not None:
            self.embeddings = np.vstack([np.array(kept_embeddings), new_embeddings])
        elif new_embeddings is not None:
            self.embeddings = new_embeddings
        elif kept_embeddings:
            self.embeddings = np.array(kept_embeddings)
        else:
            raise ValueError(f"No chunks found in corpus: {corpus_dir}")

        # Collect unique paper IDs
        paper_ids = set(c.paper_id for c in self.chunks)

        # Store metadata with paper hashes for future incremental builds
        self.metadata = {
            "num_papers": len(paper_ids),
            "num_chunks": len(self.chunks),
            "embedding_dim": embedding_adapter.dimension,
            "indexed_at": datetime.now().isoformat(),
            "corpus_path": str(corpus_dir.resolve()),
            "paper_hashes": current_hashes,
        }

        # Persist to disk
        self._save()

        return IndexStats(
            num_papers=self.metadata["num_papers"],
            num_chunks=self.metadata["num_chunks"],
            embedding_dim=self.metadata["embedding_dim"],
            indexed_at=self.metadata["indexed_at"],
            corpus_path=self.metadata["corpus_path"],
        )

    def _save(self) -> None:
        """Save index to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Save chunks
        chunks_data = [c.to_dict() for c in self.chunks]
        with open(self.index_dir / "chunks.json", "w") as f:
            json.dump(chunks_data, f)

        # Save embeddings
        if self.embeddings is not None:
            np.save(self.index_dir / "embeddings.npy", self.embeddings)

        # Save metadata
        with open(self.index_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        chunks_file = self.index_dir / "chunks.json"
        embeddings_file = self.index_dir / "embeddings.npy"
        metadata_file = self.index_dir / "metadata.json"

        if not all(f.exists() for f in [chunks_file, embeddings_file, metadata_file]):
            return False

        with open(chunks_file) as f:
            chunks_data = json.load(f)
        self.chunks = [Chunk.from_dict(d) for d in chunks_data]

        self.embeddings = np.load(embeddings_file)

        with open(metadata_file) as f:
            self.metadata = json.load(f)

        return True

    def get_stats(self) -> IndexStats | None:
        """Get index statistics."""
        if not self.metadata:
            return None
        return IndexStats(**self.metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        paper_filter: set[str] | None = None,
        tag_filter: set[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks using cosine similarity."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Compute cosine similarities
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embedding_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = embedding_norms @ query_norm

        # Create (index, score) pairs
        scored = [(i, float(similarities[i])) for i in range(len(self.chunks))]

        # Apply filters if specified
        if paper_filter:
            scored = [(i, s) for i, s in scored if self.chunks[i].paper_id in paper_filter]

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        results = []
        for i, score in scored[:top_k]:
            results.append((self.chunks[i], score))

        return results


def get_default_index_dir() -> Path:
    """Get the default index directory."""
    return Path.home() / ".zotqa" / "index"
