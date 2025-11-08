from FlagEmbedding import FlagReranker
import torch
import warnings
import os
import logging

# Suppress warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# best legal-aware reranker model
RERANK_MODEL = "BAAI/bge-reranker-base"

# Lazy initialization
_reranker = None
_reranker_failed = False  # Track if reranker failed to initialize

def _get_reranker():
    """Lazy load reranker model."""
    global _reranker, _reranker_failed
    if _reranker is None and not _reranker_failed:
        try:
            # Force CPU to avoid CUDA issues
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                _reranker = FlagReranker(RERANK_MODEL, use_fp16=False)
                if hasattr(_reranker, 'model'):
                    _reranker.model = _reranker.model.to('cpu')
        except Exception as e:
            print(f"[WARN] Failed to initialize reranker: {e}. Reranking will be disabled.")
            _reranker_failed = True
            return None
    return _reranker

def rerank(query, texts):
    """
    Rerank candidate texts using cross-encoder.
    
    Args:
        query: User's question
        texts: List of candidate texts to rerank
    
    Returns:
        List of texts sorted by relevance (most relevant first)
    """
    if not texts:
        return []
    
    # Filter out empty or None texts
    valid_texts = [t for t in texts if t and isinstance(t, str) and t.strip()]
    if not valid_texts:
        return []
    
    # Ensure query is valid
    if not query or not isinstance(query, str) or not query.strip():
        return valid_texts  # Return as-is if query is invalid
    
    # Check if reranker is available
    reranker_model = _get_reranker()
    if reranker_model is None:
        # Reranker not available, return as-is
        return valid_texts
    
    try:
        # Create pairs, ensuring both query and text are non-empty
        pairs = []
        for t in valid_texts:
            if t and t.strip() and query and query.strip():
                # Limit text length to avoid tokenizer issues (max ~512 tokens)
                text = t.strip()[:2000]  # Rough limit to avoid very long texts
                pairs.append((query.strip(), text))
        
        # Handle empty pairs
        if not pairs:
            return valid_texts
        
        # Limit number of pairs to avoid memory issues
        if len(pairs) > 50:
            pairs = pairs[:50]
            valid_texts = valid_texts[:50]
        
        # Compute scores with error handling
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                scores = reranker_model.compute_score(pairs)
        except (IndexError, ValueError, TypeError, RuntimeError) as e:
            # If reranking fails, return texts in original order
            return valid_texts
        except Exception as e:
            # Catch any other exceptions silently
            return valid_texts
        
        # Handle case where scores might be a single value or list
        if isinstance(scores, (int, float)):
            scores = [scores]
        elif not isinstance(scores, list):
            try:
                scores = list(scores)
            except (TypeError, ValueError):
                return valid_texts
        
        if len(scores) != len(pairs):
            return valid_texts
        
        # Sort by scores
        sorted_pairs = sorted(zip(scores, valid_texts[:len(scores)]), reverse=True)
        return [t for _, t in sorted_pairs]
    except Exception:
        # Silent fail - return original order
        return valid_texts
