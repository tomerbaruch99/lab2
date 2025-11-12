import warnings
import os
import logging

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Strong default cross-encoder reranker (FlagEmbedding).
# If unavailable, we gracefully fall back to passthrough.
RERANK_MODEL = "BAAI/bge-reranker-base"

_reranker = None
_reranker_failed = False

def _get_reranker():
    global _reranker, _reranker_failed
    if _reranker is None and not _reranker_failed:
        try:
            from FlagEmbedding import FlagReranker
            _reranker = FlagReranker(RERANK_MODEL, use_fp16=False)
            if hasattr(_reranker, "model"):
                _reranker.model = _reranker.model.to("cpu")
        except Exception as e:
            print(f"[WARN] Failed to initialize reranker: {e}. Reranking will be disabled.")
            _reranker_failed = True
            return None
    return _reranker

def rerank(query, texts):
    """
    Rerank candidate texts using cross-encoder.
    Returns the texts sorted by relevance (desc).
    """
    if not texts:
        return []
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not valid_texts:
        return []

    if not isinstance(query, str) or not query.strip():
        return valid_texts

    model = _get_reranker()
    if model is None:
        return valid_texts

    try:
        pairs = []
        for t in valid_texts:
            txt = t.strip()
            if not txt:
                continue
            # Truncate long texts to avoid tokenizer overflow
            txt = txt[:2000]
            pairs.append((query.strip(), txt))
        if not pairs:
            return valid_texts

        if len(pairs) > 50:
            pairs = pairs[:50]
            valid_texts = valid_texts[:50]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores = model.compute_score(pairs)

        # Normalize output shape
        try:
            scores = list(scores)
        except Exception:
            scores = [float(scores)] * len(pairs)

        if len(scores) != len(pairs):
            return valid_texts

        ranked = sorted(zip(scores, valid_texts[:len(scores)]), key=lambda x: x[0], reverse=True)
        return [t for _, t in ranked]
    except Exception:
        return valid_texts
