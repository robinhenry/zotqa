"""RAG and LLM integration."""

from zotqa.rag.chunk import Chunk, chunk_corpus, chunk_paper
from zotqa.rag.embed import (
    EmbeddingAdapter,
    MockEmbedding,
    OpenAIEmbedding,
    VoyageEmbedding,
    detect_embedding_provider,
    get_embedding_adapter,
)
from zotqa.rag.index import IndexStats, VectorIndex, get_default_index_dir
from zotqa.rag.llm import AnthropicLLM, LLMAdapter, LLMResponse, MockLLM, get_llm_adapter
from zotqa.rag.query import QueryEngine, QueryResult, UsedChunk, format_answer
from zotqa.rag.retrieve import RetrievalResult, Retriever

__all__ = [
    # Chunking
    "Chunk",
    "chunk_corpus",
    "chunk_paper",
    # Embedding
    "EmbeddingAdapter",
    "VoyageEmbedding",
    "OpenAIEmbedding",
    "MockEmbedding",
    "get_embedding_adapter",
    "detect_embedding_provider",
    # Index
    "VectorIndex",
    "IndexStats",
    "get_default_index_dir",
    # LLM
    "LLMAdapter",
    "AnthropicLLM",
    "MockLLM",
    "LLMResponse",
    "get_llm_adapter",
    # Retrieval
    "Retriever",
    "RetrievalResult",
    # Query
    "QueryEngine",
    "QueryResult",
    "UsedChunk",
    "format_answer",
]
