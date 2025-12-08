"""
Shared configuration constants for Haifa Municipality RAG system.

All hyperparameters, constants, and configurable values should be defined here
rather than hard-coded throughout the codebase.
"""

from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

# ============================================================
# API and Model Configuration
# ============================================================

# Default paths
DEFAULT_API_KEYS_PATH = "utils/api_keys.json"
DEFAULT_INDEX_NAME = "haifa-municipality-rag-index"

# Embedding model
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_DEVICE = "cpu"  # Force CPU to avoid CUDA compatibility issues

# Gemini model
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Retrieval defaults
DEFAULT_TOP_K = 5

# ============================================================
# Gemini API Configuration
# ============================================================

DEFAULT_SLEEP_BETWEEN_CALLS = 1.0  # seconds
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_RETRY_DELAY = 10.0  # seconds

# ============================================================
# Data Preparation Configuration
# ============================================================

DEFAULT_CHUNK_CHARS = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Chunking strategies
CHUNKING_STRATEGIES = ["baseline", "sentence", "adaptive"]

# ============================================================
# Namespace Detection Configuration
# ============================================================

NAMESPACE_RULES: Dict[str, List[str]] = {
    "arnona": ["ארנונה", "תשלום נכס", "חשבון", "מס", "חיוב"],
    "parking": ["חניה", "דוח", "דו\"ח", "קנס", "תווית", "תו"],
    "water": ["מים", "תאגיד", "נזילה", "חשבונית מים"],
    "sanitation": ["זבל", "אשפה", "ניקיון", "תברואה", "פינוי"],
    "welfare": ["רווחה", "שירותים חברתיים", "סיוע", "משפחה"],
    "engineering": ["היתר", "בניין", "הנדסה", "תכנון"],
    "emergency": ["מקלט", "חירום", "אזעקה", "טילים", "מלחמה"],
    "culture": ["אירוע", "תרבות", "מופע", "תערוכה", "חג"],
}

FALLBACK_NAMESPACE = "general"

# ============================================================
# Query Enhancement Configuration
# ============================================================

# Evidence detection keywords (Hebrew)
EVIDENCE_KEYWORDS = [
    "ראיות",
    "מקורות",
    "מסמכים",
    "תיעוד",
    "קובץ מקורי",
    "מקור",
    "איפה מצאת",
    "איך יודע",
    "איך את יודע",
    "הצג לי",
    "הראה לי",
    "פרטים נוספים",
    "מידע נוסף",
    "פרט יותר",
    "pdf",
]

# Evidence multiplier: if user asks for evidence, retrieve more chunks
EVIDENCE_TOP_K_MULTIPLIER = 2

# Reranking multiplier: when reranking is enabled, retrieve more chunks initially
RERANKING_TOP_K_MULTIPLIER = 2

# ============================================================
# Confidence Meter Configuration
# ============================================================

# Confidence score weights (must sum to 1.0)
CONFIDENCE_WEIGHTS = {
    "avg_chunk_similarity": 0.5,
    "retrieval_overlap": 0.3,
    "supported_claim_ratio": 0.2,
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "high": 70.0,
    "medium": 40.0,
    "low": 0.0,
}

# Similarity thresholds for confidence calculations
SIMILARITY_THRESHOLD_HIGH = 0.7
SIMILARITY_THRESHOLD_MEDIUM = 0.4
SUPPORTED_CLAIM_THRESHOLD = 0.3  # Minimum similarity to consider claim supported
SUPPORTED_CLAIM_RATIO_HIGH = 0.8
SUPPORTED_CLAIM_RATIO_MEDIUM = 0.5

# ============================================================
# Indexing Configuration
# ============================================================

DEFAULT_BATCH_SIZE = 32

# ============================================================
# Prompt Building Configuration
# ============================================================

# Default prompt style
DEFAULT_PROMPT_STYLE = "detailed"  # Options: detailed, concise, conversational, structured, eval

# ============================================================
# Evaluation Configuration
# ============================================================

DEFAULT_EVALUATION_STRATEGIES = ["baseline", "sentence", "adaptive"]
DEFAULT_EVALUATION_TOP_K = 5

# ============================================================
# Path Configuration
# ============================================================

# Data paths (relative to project root)
DEFAULT_INPUT_JSON_PATH = "scrape_and_prepare_data/haifa_scraped.json"
DEFAULT_PREPARED_DATA_DIR = "scrape_and_prepare_data/haifa_prepared_data"
DEFAULT_OUTPUT_PARQUET_NAME = "haifa_rag_chunks.parquet"
DEFAULT_OUTPUT_CSV_NAME = "haifa_rag_chunks.csv"
DEFAULT_PAGE_INDEX_PATH = "scrape_and_prepare_data/page_index.csv"

# Evaluation paths
DEFAULT_EVALUATION_OUTPUT_DIR = "evaluation/evaluation_results"
DEFAULT_EVALUATION_QUERIES_PATH = "evaluation/evaluation_queries.json"
DEFAULT_TESTSET_PATH = "tests/embedding_testset.json"

# ============================================================
# Configuration Classes (for structured config)
# ============================================================

@dataclass
class GeminiConfig:
    """Configuration for Gemini API calls."""
    model_name: str = DEFAULT_GEMINI_MODEL
    sleep_between_calls: float = DEFAULT_SLEEP_BETWEEN_CALLS
    max_retries: int = DEFAULT_MAX_RETRIES
    initial_retry_delay: float = DEFAULT_INITIAL_RETRY_DELAY


@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations."""
    top_k: int = DEFAULT_TOP_K
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    index_name: str = DEFAULT_INDEX_NAME
    device: str = DEFAULT_EMBEDDING_DEVICE


@dataclass
class ChunkingConfig:
    """Configuration for data chunking."""
    chunk_chars: int = DEFAULT_CHUNK_CHARS
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    strategies: List[str] = None
    
    def __post_init__(self):
        if self.strategies is None:
            self.strategies = CHUNKING_STRATEGIES.copy()


@dataclass
class ConfidenceConfig:
    """Configuration for confidence scoring."""
    weights: Dict[str, float] = None
    thresholds: Dict[str, float] = None
    similarity_threshold_high: float = SIMILARITY_THRESHOLD_HIGH
    similarity_threshold_medium: float = SIMILARITY_THRESHOLD_MEDIUM
    supported_claim_threshold: float = SUPPORTED_CLAIM_THRESHOLD
    supported_claim_ratio_high: float = SUPPORTED_CLAIM_RATIO_HIGH
    supported_claim_ratio_medium: float = SUPPORTED_CLAIM_RATIO_MEDIUM
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = CONFIDENCE_WEIGHTS.copy()
        if self.thresholds is None:
            self.thresholds = CONFIDENCE_THRESHOLDS.copy()


@dataclass
class PathConfig:
    """Configuration for file paths."""
    api_keys_path: str = DEFAULT_API_KEYS_PATH
    input_json_path: str = DEFAULT_INPUT_JSON_PATH
    prepared_data_dir: str = DEFAULT_PREPARED_DATA_DIR
    page_index_path: str = DEFAULT_PAGE_INDEX_PATH
    evaluation_output_dir: str = DEFAULT_EVALUATION_OUTPUT_DIR
    evaluation_queries_path: str = DEFAULT_EVALUATION_QUERIES_PATH
    testset_path: str = DEFAULT_TESTSET_PATH


# ============================================================
# Default Configuration Instances
# ============================================================

# Create default config instances for convenience
DEFAULT_GEMINI_CONFIG = GeminiConfig()
DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()
DEFAULT_CHUNKING_CONFIG = ChunkingConfig()
DEFAULT_CONFIDENCE_CONFIG = ConfidenceConfig()
DEFAULT_PATH_CONFIG = PathConfig()
