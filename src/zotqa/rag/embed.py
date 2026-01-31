"""Embedding adapter for computing text embeddings."""

import os
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm


class EmbeddingAdapter(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed a list of texts, returning an array of shape (n_texts, dim)."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query, returning an array of shape (dim,)."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class VoyageEmbedding(EmbeddingAdapter):
    """Voyage AI embedding adapter."""

    def __init__(self, model: str = "voyage-3-lite", api_key: str | None = None):
        try:
            import voyageai
        except ImportError:
            raise ImportError("voyageai not installed. Install with: pip install voyageai")

        self.model = model
        self.client = voyageai.Client(api_key=api_key or os.environ.get("VOYAGE_API_KEY"))
        self._dimension = 512 if "lite" in model else 1024

    def embed(self, texts: list[str], batch_size: int = 128, show_progress: bool = False) -> np.ndarray:
        """Embed texts in batches."""
        all_embeddings = []
        batches = range(0, len(texts), batch_size)
        if show_progress:
            batches = tqdm(list(batches), desc="Embedding", unit="batch")
        for i in batches:
            batch = texts[i : i + batch_size]
            result = self.client.embed(batch, model=self.model, input_type="document")
            all_embeddings.extend(result.embeddings)
        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        result = self.client.embed([query], model=self.model, input_type="query")
        return np.array(result.embeddings[0], dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbedding(EmbeddingAdapter):
    """OpenAI embedding adapter."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")

        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        # text-embedding-3-small = 1536, text-embedding-3-large = 3072
        self._dimension = 1536 if "small" in model else 3072

    def embed(self, texts: list[str], batch_size: int = 2048, show_progress: bool = False) -> np.ndarray:
        """Embed texts in batches."""
        all_embeddings = []
        batches = range(0, len(texts), batch_size)
        if show_progress:
            batches = tqdm(list(batches), desc="Embedding", unit="batch")
        for i in batches:
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            for item in response.data:
                all_embeddings.append(item.embedding)
        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        response = self.client.embeddings.create(input=[query], model=self.model)
        return np.array(response.data[0].embedding, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension


class MockEmbedding(EmbeddingAdapter):
    """Mock embedding for testing (random vectors)."""

    def __init__(self, dimension: int = 128):
        self._dimension = dimension
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        embeddings = []
        text_iter = texts
        if show_progress:
            text_iter = tqdm(texts, desc="Embedding", unit="text")
        for text in text_iter:
            if text not in self._cache:
                # Use hash for reproducibility
                seed = hash(text) % (2**32)
                rng = np.random.default_rng(seed)
                vec = rng.standard_normal(self._dimension).astype(np.float32)
                vec = vec / np.linalg.norm(vec)  # Normalize
                self._cache[text] = vec
            embeddings.append(self._cache[text])
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_adapter(provider: str = "voyage", **kwargs) -> EmbeddingAdapter:
    """Factory function to get an embedding adapter.

    Args:
        provider: One of "voyage", "openai", or "mock"
        **kwargs: Provider-specific arguments (model, api_key, etc.)

    Returns:
        An embedding adapter instance
    """
    if provider == "voyage":
        return VoyageEmbedding(**kwargs)
    elif provider == "openai":
        return OpenAIEmbedding(**kwargs)
    elif provider == "mock":
        return MockEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def detect_embedding_provider() -> str:
    """Detect available embedding provider based on installed packages and API keys."""
    # Check for Voyage
    if os.environ.get("VOYAGE_API_KEY"):
        try:
            import importlib.util

            if importlib.util.find_spec("voyageai"):
                return "voyage"
        except ImportError:
            pass

    # Check for OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import importlib.util

            if importlib.util.find_spec("openai"):
                return "openai"
        except ImportError:
            pass

    raise RuntimeError(
        "No embedding provider available. Set VOYAGE_API_KEY or OPENAI_API_KEY "
        "and install the corresponding package (voyageai or openai)."
    )
