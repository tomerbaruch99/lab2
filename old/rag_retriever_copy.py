import os
import re
import logging
import warnings
from typing import Optional, List, Dict, Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from bm25_index_copy import bm25_search
from reranker_copy import rerank
from query_understanding_copy import map_query_to_category

# Suppress extra logs
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Lazy globals
_query_encoder = None
_clauses = None
_emb = None
_faiss_index = None

def _initialize_models():
    """
    Loads:
      - Jina encoder (CPU)
      - CUAD long clauses parquet
      - Precomputed normalized embeddings (.npy)
      - FAISS IP index (cosine over normalized vectors)
    """
    global _query_encoder, _clauses, _emb, _faiss_index

    if _query_encoder is None:
        _query_encoder = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", device="cpu")

    if _clauses is None:
        _clauses = pd.read_parquet("cuad_prepared_data/cuad_long_clauses.parquet")

    if _emb is None:
        _emb = np.load("processed_data/dense_embeddings.npy", mmap_mode="r").astype("float32")
        # _emb = np.load("processed_data/jina_embeddings.npy").astype("float32")

    if _faiss_index is None:
        _faiss_index = faiss.IndexFlatIP(_emb.shape[1])  # cosine via IP on normalized vectors
        _faiss_index.add(_emb)

    return _query_encoder, _clauses, _emb, _faiss_index

def _apply_simple_filename_filter(df: pd.DataFrame, filename: Optional[str]) -> pd.DataFrame:
    if not filename:
        return df
    needle = str(filename).lower().strip()
    return df[df["filename"].str.lower().str.contains(needle, na=False)]

def _apply_category_filter(df: pd.DataFrame, category: Optional[str]) -> pd.DataFrame:
    if not category:
        return df
    return df[df["category"].str.lower() == str(category).lower()]

def retrieve(query: str,
             k_dense: int = 8,
             k_bm25: int = 8,
             final_k: int = 5,
             filename: Optional[str] = None,
             category: Optional[str] = None,
             use_query_understanding: bool = True,
             confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval (Dense + BM25) with simple metadata filters and cross-encoder reranking.
    Returns a list of dicts including text, filename, category, answer, question_template.
    """
    query_encoder, clauses, emb, index = _initialize_models()

    # Optional: map query to category
    if use_query_understanding and not category:
        cat, conf = map_query_to_category(query, confidence_threshold)
        if cat:
            category = cat
            print(f"[Query Understanding] Mapped to category: {category} (confidence: {conf:.2f})")

    # BM25 candidates (already filtered by filename/category inside bm25_search)
    bm25_df = bm25_search(query, k=k_bm25, filename=filename, category=category)
    bm25_indices = bm25_df.index.tolist()

    # Dense search (global index), then local filter
    qvec = query_encoder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    sims, idx = index.search(qvec, k_dense * 5)  # oversample to give filters a chance
    dense_pool = idx[0].tolist()

    def _keep(i):
        if filename:
            if str(filename).lower() not in str(clauses.at[i, "filename"]).lower():
                return False
        if category:
            if str(clauses.at[i, "category"]).lower() != str(category).lower():
                return False
        return True

    dense_indices = [i for i in dense_pool if _keep(i)][:k_dense]

    # Merge unique (order-preserving)
    candidate_indices = list(dict.fromkeys(dense_indices + bm25_indices))
    if not candidate_indices:
        print("[INFO] No candidates found.")
        return []

    cand_df = clauses.iloc[candidate_indices].copy()
    cand_texts = cand_df["context"].fillna("").astype(str).tolist()

    # Rerank texts (keep index mapping)
    reranked_texts = rerank(query, cand_texts)

    # Map back by first occurrence
    text_to_first_idx = {}
    for i, t in enumerate(cand_texts):
        text_to_first_idx.setdefault(t, candidate_indices[i])

    results = []
    for t in reranked_texts[:final_k]:
        ridx = text_to_first_idx.get(t, None)
        if ridx is None:
            results.append({
                "text": t, "filename": None, "category": None, "answer": "", "question_template": ""
            })
            continue
        row = clauses.iloc[ridx]
        results.append({
            "text": t,
            "filename": row.get("filename", None),
            "category": row.get("category", None),
            "answer": row.get("answer", ""),
            "question_template": row.get("question_template", ""),
            "answer_type": row.get("answer_type", ""),        # NEW (optional)
            "contract_type": row.get("contract_type", ""),    # NEW (optional)
            "split": row.get("split", "")                     # NEW (optional)
        })

    return results
