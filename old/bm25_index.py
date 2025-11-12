from rank_bm25 import BM25Okapi
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional

BM25_INDEX_FILE = "processed_data/bm25_index.pkl"
BM25_DF_FILE = "processed_data/bm25_df.pkl"

# Global cache
_cached_df = None
_cached_bm25 = None

def build_bm25(input_parquet="cuad_prepared_data/cuad_long_clauses.parquet", 
               save_cache=True):
    """Build BM25 index and optionally cache it."""
    df = pd.read_parquet(input_parquet)
    tokenized = [t.lower().split() for t in df["context"]]
    bm25 = BM25Okapi(tokenized)
    
    if save_cache:
        Path(BM25_INDEX_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_INDEX_FILE, "wb") as f:
            pickle.dump(bm25, f)
        df.to_pickle(BM25_DF_FILE)
        print(f"[✓] BM25 index cached to {BM25_INDEX_FILE}")
    
    return df, bm25

def load_bm25():
    """Load cached BM25 index and dataframe."""
    global _cached_df, _cached_bm25
    
    if _cached_df is not None and _cached_bm25 is not None:
        return _cached_df, _cached_bm25
    
    if Path(BM25_INDEX_FILE).exists() and Path(BM25_DF_FILE).exists():
        with open(BM25_INDEX_FILE, "rb") as f:
            _cached_bm25 = pickle.load(f)
        _cached_df = pd.read_pickle(BM25_DF_FILE)
        return _cached_df, _cached_bm25
    else:
        # Build and cache if not exists
        _cached_df, _cached_bm25 = build_bm25()
        return _cached_df, _cached_bm25

def bm25_search(query, k=10, 
                filename: Optional[str] = None,
                category: Optional[str] = None,
                input_parquet="cuad_prepared_data/cuad_long_clauses.parquet"):
    """
    BM25 search with optional metadata filtering.
    
    Args:
        query: Search query
        k: Number of results to return
        filename: Filter by filename (optional)
        category: Filter by category (optional)
        input_parquet: Path to parquet file (used if cache doesn't exist)
    
    Returns:
        DataFrame with top-k results
    """
    df, bm25 = load_bm25()
    
    # Apply metadata filters before searching
    filtered_df = df.copy()
    if filename is not None:
        filtered_df = filtered_df[filtered_df["filename"] == filename]
    if category is not None:
        filtered_df = filtered_df[filtered_df["category"] == category]
    
    if len(filtered_df) == 0:
        return pd.DataFrame()
    
    # Get indices in original dataframe
    filtered_indices = filtered_df.index.tolist()
    
    # Compute scores only for filtered documents
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    
    # Filter scores to only include filtered documents
    filtered_scores = scores[filtered_indices]
    
    # Get top-k from filtered results
    top_local_indices = filtered_scores.argsort()[-k:][::-1]
    top_global_indices = [filtered_indices[i] for i in top_local_indices]
    
    return df.iloc[top_global_indices]

if __name__ == "__main__":
    # Build index on first run
    # build_bm25()
    
    # Then use cached version
    print(bm25_search("non compete", k=5))


# # faster version by the GPT
#     from rank_bm25 import BM25Okapi
# import pandas as pd
# import pickle

# BM25_INDEX_FILE = "processed_data/bm25_index.pkl"
# BM25_DF_FILE = "processed_data/bm25_df.pkl"

# def build_bm25(input_parquet="cuad_prepared_data/cuad_long_clauses.parquet"):
#     df = pd.read_parquet(input_parquet)
#     tokenized = [t.lower().split() for t in df["context"]]
#     bm25 = BM25Okapi(tokenized)

#     with open(BM25_INDEX_FILE, "wb") as f:
#         pickle.dump(bm25, f)
#     df.to_pickle(BM25_DF_FILE)

#     print("✅ BM25 index saved.")
#     return df, bm25

# def load_bm25():
#     df = pd.read_pickle(BM25_DF_FILE)
#     with open(BM25_INDEX_FILE, "rb") as f:
#         bm25 = pickle.load(f)
#     return df, bm25

# def bm25_search(query, k=10):
#     df, bm25 = load_bm25()
#     scores = bm25.get_scores(query.lower().split())
#     top_indices = scores.argsort()[-k:][::-1]
#     return df.iloc[top_indices]

# if __name__ == "__main__":
#     # Run this ONCE to create the index
#     # build_bm25()

#     # Then use this for fast lookup (no reprocessing)
#     print(bm25_search("non compete"))
