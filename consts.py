import os

import os
import argparse

def load_experiment_config():
    """
    Reads default values (from config.py) and allows CLI overrides.
    Returns a dict of runtime configuration flags.
    """
    parser = argparse.ArgumentParser(description="CUAD RAG runtime config")
    
    # category router
    parser.add_argument("--category_router_mode", default="off", choices=["off", "hard", "soft"],
                        help="Enable category-aware search: off|hard|soft")
    parser.add_argument("--category_hard_bonus", type=float, default=0.20)
    parser.add_argument("--category_soft_max_bonus", type=float, default=0.12)
    parser.add_argument("--category_query_weight", type=float, default=1.0)
    parser.add_argument("--category_max_reps", type=int, default=3)
    parser.add_argument("--filename_match_bonus", type=float, default=0.03)

    # reranker
    parser.add_argument("--reranker", default="", help="cross-encoder model; empty disables")

    # embeddings
    parser.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=8)

    # evaluation
    parser.add_argument("--eval_provider", default="gemini")
    parser.add_argument("--eval_model", default="gemini-1.5-flash")

    args, unknown = parser.parse_known_args()
    return vars(args)


CUAD_CSV_FILEPATH = "data/CUAD_v1/master_clauses.csv"
CUAD_JSON_FILEPATH = "data/CUAD_v1/CUAD_v1.json"
FULL_CONTRACTS_TXT_DIR = "data/CUAD_v1/full_contract_txt/"

# Paths (point to your actual ETL out_dir)
CUAD_OUT_DIR = os.getenv("CUAD_OUT_DIR", "./cuad_prepared_data")
LONG_PARQUET = os.path.join(CUAD_OUT_DIR, "cuad_long_clauses.parquet")
LONG_JSONL   = os.path.join(CUAD_OUT_DIR, "cuad_long_clauses.jsonl")
PARA_INDEX_PARQUET = os.path.join(CUAD_OUT_DIR, "cuad_paragraph_index.parquet")  # optional

# Embeddings / Reranker
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Good legal alternative: "intfloat/e5-base-v2" (prefix queries with “query: ” if you use E5) or "nomic-ai/nomic-embed-text-v1.5" (+ bge reranker).
RERANKER = os.getenv("RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")  # set "" to disable

# Vector store choice
VSTORE = os.getenv("VSTORE", "faiss")  # "pinecone" or "faiss"

# Pinecone
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "cuad-contracts")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

# Retrieval defaults
TOP_K = int(os.getenv("TOP_K", 8))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 2800))  # trimming packed context

# LLM provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "openai" | "gemini" | "cohere" | "qwen"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")  # adapt to your account

# Synthetic QA generation config
SYNTH_PROVIDER = os.getenv("SYNTH_PROVIDER", "gemini")  # "gemini" | "openai" | "cohere"
SYNTH_MODEL    = os.getenv("SYNTH_MODEL",   "gemini-1.5-flash")  # e.g., "gpt-4o-mini", "command-r-plus"
SYNTH_MAX_PER_CATEGORY = int(os.getenv("SYNTH_MAX_PER_CATEGORY", "25"))
SYNTH_TEMP = float(os.getenv("SYNTH_TEMP", "0.2"))

# LLM rubric evaluator
EVAL_PROVIDER = os.getenv("EVAL_PROVIDER", "gemini")  # "gemini" | "openai" | "cohere"
EVAL_MODEL    = os.getenv("EVAL_MODEL",   "gemini-1.5-flash")
EVAL_TEMP     = float(os.getenv("EVAL_TEMP", "0.0"))

# Category-aware search
# Modes: "off" (no bias), "hard" (strong bias), "soft" (dynamic/gentle bias)
CATEGORY_ROUTER_MODE = os.getenv("CATEGORY_ROUTER_MODE", "off")  # "off" | "hard" | "soft"

# Score bonuses applied to initial dense scores and (if present) reranker scores
CATEGORY_HARD_BONUS = float(os.getenv("CATEGORY_HARD_BONUS", "0.20"))  # fixed bonus when category matches
CATEGORY_SOFT_MAX_BONUS = float(os.getenv("CATEGORY_SOFT_MAX_BONUS", "0.12"))  # max bonus * confidence

# Query hint strength: how much to push the category into the query text
# In "hard" mode we repeat the hint up to MAX reps; "soft" scales by confidence.
CATEGORY_QUERY_WEIGHT = float(os.getenv("CATEGORY_QUERY_WEIGHT", "1.0"))  # text hint multiplier
CATEGORY_MAX_REPS = int(os.getenv("CATEGORY_MAX_REPS", "3"))  # cap hint repetitions

# Optional: also reward filename matches slightly (helps when user mentions file/party)
FILENAME_MATCH_BONUS = float(os.getenv("FILENAME_MATCH_BONUS", "0.03"))

"""
Bonuses too strong? Lower CATEGORY_HARD_BONUS or CATEGORY_SOFT_MAX_BONUS.
Query drift? Reduce CATEGORY_MAX_REPS or CATEGORY_QUERY_WEIGHT.
Redactions preserved? Already handled by your ETL and prompt; no changes needed.
Latency sensitive? Keep reranker on but reduce overfetch (e.g., from k*3 to k*2) when using category bias.
"""

