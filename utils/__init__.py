"""
Shared utilities for Haifa Municipality RAG system.
"""

from .config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_NAME,
    DEFAULT_API_KEYS_PATH,
    DEFAULT_TOP_K,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_SLEEP_BETWEEN_CALLS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_INITIAL_RETRY_DELAY,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    NAMESPACE_RULES,
    FALLBACK_NAMESPACE,
    EVIDENCE_KEYWORDS,
    EVIDENCE_TOP_K_MULTIPLIER,
    RERANKING_TOP_K_MULTIPLIER,
    CONFIDENCE_WEIGHTS,
    CONFIDENCE_THRESHOLDS,
    SIMILARITY_THRESHOLD_HIGH,
    SIMILARITY_THRESHOLD_MEDIUM,
    SUPPORTED_CLAIM_THRESHOLD,
    SUPPORTED_CLAIM_RATIO_HIGH,
    SUPPORTED_CLAIM_RATIO_MEDIUM,
    # Config classes
    GeminiConfig,
    RetrievalConfig,
    ChunkingConfig,
    ConfidenceConfig,
    PathConfig,
    DEFAULT_GEMINI_CONFIG,
    DEFAULT_RETRIEVAL_CONFIG,
    DEFAULT_CHUNKING_CONFIG,
    DEFAULT_CONFIDENCE_CONFIG,
    DEFAULT_PATH_CONFIG,
    # Evaluation configuration
    DEFAULT_EVALUATION_STRATEGIES,
    DEFAULT_EVALUATION_TOP_K,
    # Indexing configuration
    DEFAULT_BATCH_SIZE,
    # Embedding configuration
    DEFAULT_EMBEDDING_DEVICE,
    # Chunking configuration
    CHUNKING_STRATEGIES,
    # Prompt configuration
    DEFAULT_PROMPT_STYLE,
    # Path configuration
    DEFAULT_INPUT_JSON_PATH,
    DEFAULT_PREPARED_DATA_DIR,
    DEFAULT_OUTPUT_PARQUET_NAME,
    DEFAULT_OUTPUT_CSV_NAME,
    DEFAULT_PAGE_INDEX_PATH,
    DEFAULT_EVALUATION_OUTPUT_DIR,
    DEFAULT_EVALUATION_QUERIES_PATH,
    DEFAULT_TESTSET_PATH,
)

from .pinecone_utils import (
    load_pinecone_api_key,
    create_index,
)

from .embedding import EmbeddingModel

__all__ = [
    # Constants
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_INDEX_NAME",
    "DEFAULT_API_KEYS_PATH",
    "DEFAULT_TOP_K",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_SLEEP_BETWEEN_CALLS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_INITIAL_RETRY_DELAY",
    "DEFAULT_CHUNK_CHARS",
    "DEFAULT_CHUNK_OVERLAP",
    "NAMESPACE_RULES",
    "FALLBACK_NAMESPACE",
    "EVIDENCE_KEYWORDS",
    "EVIDENCE_TOP_K_MULTIPLIER",
    "RERANKING_TOP_K_MULTIPLIER",
    "CONFIDENCE_WEIGHTS",
    "CONFIDENCE_THRESHOLDS",
    "SIMILARITY_THRESHOLD_HIGH",
    "SIMILARITY_THRESHOLD_MEDIUM",
    "SUPPORTED_CLAIM_THRESHOLD",
    "SUPPORTED_CLAIM_RATIO_HIGH",
    "SUPPORTED_CLAIM_RATIO_MEDIUM",
    # Config classes
    "GeminiConfig",
    "RetrievalConfig",
    "ChunkingConfig",
    "ConfidenceConfig",
    "PathConfig",
    "DEFAULT_GEMINI_CONFIG",
    "DEFAULT_RETRIEVAL_CONFIG",
    "DEFAULT_CHUNKING_CONFIG",
    "DEFAULT_CONFIDENCE_CONFIG",
    "DEFAULT_PATH_CONFIG",
    # Evaluation configuration
    "DEFAULT_EVALUATION_STRATEGIES",
    "DEFAULT_EVALUATION_TOP_K",
    # Indexing configuration
    "DEFAULT_BATCH_SIZE",
    # Embedding configuration
    "DEFAULT_EMBEDDING_DEVICE",
    # Chunking configuration
    "CHUNKING_STRATEGIES",
    # Prompt configuration
    "DEFAULT_PROMPT_STYLE",
    # Path configuration
    "DEFAULT_INPUT_JSON_PATH",
    "DEFAULT_PREPARED_DATA_DIR",
    "DEFAULT_OUTPUT_PARQUET_NAME",
    "DEFAULT_OUTPUT_CSV_NAME",
    "DEFAULT_PAGE_INDEX_PATH",
    "DEFAULT_EVALUATION_OUTPUT_DIR",
    "DEFAULT_EVALUATION_QUERIES_PATH",
    "DEFAULT_TESTSET_PATH",
    # Utilities
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

