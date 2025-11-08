import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from reranker import rerank
from bm25_index import bm25_search
from query_understanding import map_query_to_category
from typing import Optional, List, Dict, Any
import faiss
import warnings
import os
import logging
import re

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Initialize models and indices (lazy loading)
_query_encoder = None  # Use same model as embeddings (Jina)
_clauses = None
_emb = None
_faiss_index = None

def _initialize_models():
    """Lazy initialization of models and indices."""
    global _query_encoder, _clauses, _emb, _faiss_index
    
    # Use Jina embeddings for query encoding to match stored embeddings
    if _query_encoder is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _query_encoder = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", device="cpu")
    
    if _clauses is None:
        _clauses = pd.read_parquet("cuad_prepared_data/cuad_long_clauses.parquet")
    
    if _emb is None:
        _emb = np.load("processed_data/jina_embeddings.npy").astype("float32")
    
    if _faiss_index is None:
        _faiss_index = faiss.IndexFlatL2(_emb.shape[1])
        _faiss_index.add(_emb)
    
    return _query_encoder, _clauses, _emb, _faiss_index

def retrieve(query: str, 
             k_dense: int = 8, 
             k_bm25: int = 8, 
             final_k: int = 5,
             filename: Optional[str] = None,
             category: Optional[str] = None,
             use_query_understanding: bool = True,
             confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Dual-stage retrieval with metadata filtering and query understanding.
    
    Args:
        query: User's question
        k_dense: Number of dense retrieval results
        k_bm25: Number of BM25 retrieval results
        final_k: Final number of results after reranking
        filename: Filter by filename (optional)
        category: Filter by category (optional, overrides query understanding)
        use_query_understanding: Whether to use query understanding to infer category
        confidence_threshold: Minimum confidence for query understanding
    
    Returns:
        List of dictionaries with 'text', 'filename', 'category', 'score' keys
    """
    query_encoder, clauses, emb, index = _initialize_models()
    
    # Query understanding: map query to category if not provided
    inferred_category = None
    if use_query_understanding and category is None:
        inferred_category, confidence = map_query_to_category(query, confidence_threshold)
        if inferred_category:
            category = inferred_category
            print(f"[Query Understanding] Mapped to category: {category} (confidence: {confidence:.2f})")
    
    # Apply metadata filters
    filtered_clauses = clauses.copy()
    
    if filename is not None:
        # Normalize filename for matching (lowercase, remove path, handle variations)
        filename_normalized = filename.lower().strip()
        # Remove path if present
        if '/' in filename_normalized:
            filename_normalized = filename_normalized.split('/')[-1]
        # Remove .txt extension if present
        if filename_normalized.endswith('.txt'):
            filename_normalized = filename_normalized[:-4]
        
        # Normalize further: remove extra spaces, normalize underscores/hyphens
        filename_normalized = re.sub(r'\s+', ' ', filename_normalized)  # Normalize spaces
        filename_normalized_clean = re.sub(r'[_\-\s]', '', filename_normalized)  # Remove separators for fuzzy match
        
        # Debug: Check what filenames are actually in the database
        available_filenames = clauses["filename"].unique()
        filename_lower_map = {f.lower(): f for f in available_filenames}
        
        # Try exact match first (case-sensitive) - search in clauses, not filtered_clauses
        exact_match = clauses[clauses["filename"] == filename]
        if len(exact_match) > 0:
            filtered_clauses = exact_match
            print(f"[INFO] Found {len(filtered_clauses)} clauses using exact filename match")
        
        # Try case-insensitive exact match
        if len(filtered_clauses) == 0:
            if filename_normalized in filename_lower_map:
                actual_filename = filename_lower_map[filename_normalized]
                filtered_clauses = clauses[clauses["filename"] == actual_filename]
                print(f"[INFO] Found {len(filtered_clauses)} clauses using case-insensitive match (actual: '{actual_filename}')")
        
        # Try partial match (filename contains the search term)
        if len(filtered_clauses) == 0:
            mask = clauses["filename"].str.lower().str.contains(filename_normalized, na=False, regex=False)
            if mask.any():
                filtered_clauses = clauses[mask]
                matched_filenames = clauses[mask]["filename"].unique()
                print(f"[INFO] Found {len(filtered_clauses)} clauses using partial filename match")
                print(f"[INFO] Matched filenames: {list(matched_filenames)}")
        
        # Try fuzzy match (remove separators and compare)
        if len(filtered_clauses) == 0:
            # Create normalized versions for comparison
            filename_map = {}
            for f in available_filenames:
                f_clean = re.sub(r'[_\-\s]', '', f.lower())
                filename_map[f] = f_clean
            
            # Find matches where normalized versions are similar
            matches = []
            for f, f_clean in filename_map.items():
                # Check if search term is contained in filename or vice versa
                if filename_normalized_clean in f_clean or f_clean in filename_normalized_clean:
                    matches.append(f)
                # Also check if key parts match (first part, company name, etc.)
                elif len(filename_normalized_clean) > 10:  # Only for longer filenames
                    # Check if first 10 chars match
                    if filename_normalized_clean[:10] in f_clean[:50]:
                        matches.append(f)
            
            if matches:
                filtered_clauses = clauses[clauses["filename"].isin(matches)]
                print(f"[INFO] Found {len(filtered_clauses)} clauses using fuzzy filename match")
                print(f"[INFO] Matched filenames: {matches}")
        
        if len(filtered_clauses) == 0:
            # Show similar filenames to help user
            # Find most similar filenames
            similar = []
            search_parts = set(re.sub(r'[_\-\s]', ' ', filename_normalized).split())
            
            for f in available_filenames:
                f_parts = set(re.sub(r'[_\-\s]', ' ', f.lower()).split())
                # Count matching words
                common = search_parts.intersection(f_parts)
                if len(common) >= 2:  # At least 2 words in common
                    similar.append((f, len(common)))
            
            similar.sort(key=lambda x: x[1], reverse=True)
            similar = [f for f, _ in similar[:5]]
            
            print(f"[INFO] No clauses found for filename '{filename}'")
            print(f"[INFO] Search normalized to: '{filename_normalized}'")
            if similar:
                print(f"[INFO] Similar filenames found:")
                for f in similar:
                    print(f"    - {f}")
            else:
                print(f"[INFO] Available filenames (showing first 10): {list(available_filenames[:10])}")
            print(f"[INFO] Searching all files instead")
            filtered_clauses = clauses.copy()
    
    if category is not None:
        # Apply category filter to already filtered clauses (respects filename filter)
        filtered_clauses = filtered_clauses[filtered_clauses["category"] == category]
        if len(filtered_clauses) == 0:
            # Try case-insensitive match on already filtered clauses (before category filter)
            # Need to reapply filename filter first if it was set
            if filename is not None:
                # Recreate filtered_clauses with filename filter
                filename_normalized = filename.lower().strip()
                if '/' in filename_normalized:
                    filename_normalized = filename_normalized.split('/')[-1]
                if filename_normalized.endswith('.txt'):
                    filename_normalized = filename_normalized[:-4]
                filename_normalized = re.sub(r'\s+', ' ', filename_normalized)
                filename_lower_map = {f.lower(): f for f in clauses["filename"].unique()}
                if filename_normalized in filename_lower_map:
                    actual_filename = filename_lower_map[filename_normalized]
                    temp_filtered = clauses[clauses["filename"] == actual_filename]
                else:
                    mask = clauses["filename"].str.lower().str.contains(filename_normalized, na=False, regex=False)
                    temp_filtered = clauses[mask] if mask.any() else clauses.copy()
            else:
                temp_filtered = clauses.copy()
            
            # Now try case-insensitive category match
            filtered_clauses = temp_filtered[temp_filtered["category"].str.lower() == category.lower()]
            if len(filtered_clauses) == 0:
                # If still no match, try on all clauses but warn
                filtered_clauses = clauses[clauses["category"].str.lower() == category.lower()]
                if len(filtered_clauses) == 0:
                    print(f"[INFO] No clauses found for category '{category}', searching all categories")
                    # Keep filename filter if it was set
                    if filename is not None:
                        # Reapply filename filter
                        filename_normalized = filename.lower().strip()
                        if '/' in filename_normalized:
                            filename_normalized = filename_normalized.split('/')[-1]
                        if filename_normalized.endswith('.txt'):
                            filename_normalized = filename_normalized[:-4]
                        filename_normalized = re.sub(r'\s+', ' ', filename_normalized)
                        mask = clauses["filename"].str.lower().str.contains(filename_normalized, na=False, regex=False)
                        if mask.any():
                            filtered_clauses = clauses[mask]
                        else:
                            filtered_clauses = clauses.copy()
                    else:
                        filtered_clauses = clauses.copy()
                else:
                    print(f"[INFO] Found {len(filtered_clauses)} clauses for category '{category}' (case-insensitive match, ignoring filename filter)")
            else:
                print(f"[INFO] Found {len(filtered_clauses)} clauses for category '{category}' (case-insensitive match)")
        else:
            print(f"[INFO] Found {len(filtered_clauses)} clauses for category '{category}'")
    
    if len(filtered_clauses) == 0:
        print("[INFO] No clauses available, returning empty results")
        return []
    
    # Get filtered indices
    filtered_indices = filtered_clauses.index.tolist()
    
    # Dense retrieval (using same model as stored embeddings)
    qvec = query_encoder.encode([query], convert_to_numpy=True)
    dist, idx = index.search(qvec, k_dense * 2)  # Get more to account for filtering
    
    # Filter dense results to only include filtered indices
    dense_indices = [i for i in idx[0] if i in filtered_indices][:k_dense]
    if len(dense_indices) < k_dense:
        # If not enough filtered results, get more from all
        dist, idx = index.search(qvec, k_dense * 3)
        dense_indices = [i for i in idx[0] if i in filtered_indices][:k_dense]
    
    dense_results_df = clauses.iloc[dense_indices]
    
    # BM25 retrieval (with metadata filtering)
    sparse_results_df = bm25_search(query, k=k_bm25, filename=filename, category=category)
    
    # Merge candidates (preserve metadata)
    sparse_indices = sparse_results_df.index.tolist() if len(sparse_results_df) > 0 else []
    all_indices = list(set(dense_indices + sparse_indices))
    
    if not all_indices:
        print("[WARN] No candidates found. Returning empty results.")
        return []
    
    candidates_df = clauses.iloc[all_indices]
    candidates = candidates_df["context"].tolist()
    
    # Filter out empty or invalid candidates
    candidates = [c for c in candidates if c and isinstance(c, str) and c.strip()]
    
    if not candidates:
        print("[WARN] No valid candidates after filtering. Returning empty results.")
        return []
    
    # Rerank final list
    ranked_texts = rerank(query, candidates)
    
    # Build results with metadata
    results = []
    for text in ranked_texts[:final_k]:
        # Find the row(s) matching this text
        matching_rows = candidates_df[candidates_df["context"] == text]
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            results.append({
                "text": text,
                "filename": row["filename"],
                "category": row["category"],
                "answer": row.get("answer", ""),
                "question_template": row.get("question_template", ""),
            })
        else:
            # Fallback if exact match not found
            results.append({
                "text": text,
                "filename": None,
                "category": None,
                "answer": "",
                "question_template": "",
            })
    
    return results
