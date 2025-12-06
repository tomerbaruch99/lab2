"""
Shared utilities for Haifa Municipality RAG system.
"""

from .config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_NAME,
    DEFAULT_API_KEYS_PATH,
    DEFAULT_TOP_K,
)

from .pinecone_utils import (
    load_pinecone_api_key,
    create_index,
)

from .embedding import EmbeddingModel

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_INDEX_NAME",
    "DEFAULT_API_KEYS_PATH",
    "DEFAULT_TOP_K",
    "load_pinecone_api_key",
    "create_index",
    "EmbeddingModel",
]

# Optional: Import query enhancement utilities if needed
try:
    from .query_enhancement import rephrase_query, enrich_query, rerank_chunks
    __all__.extend(["rephrase_query", "enrich_query", "rerank_chunks"])
except ImportError:
    pass

