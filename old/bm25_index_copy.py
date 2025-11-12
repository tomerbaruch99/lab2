import re
import pandas as pd
from rank_bm25 import BM25Okapi

_DF = None
_BM25 = None

def _tokenize(s: str):
    s = (s or "").lower()
    return re.findall(r"[a-z0-9]+", s)

def _init(input_parquet="cuad_prepared_data/cuad_long_clauses.parquet"):
    global _DF, _BM25
    if _DF is None:
        _DF = pd.read_parquet(input_parquet)
    if _BM25 is None:
        corpus = _DF["context"].fillna("").astype(str).tolist()
        tokenized = [_tokenize(c) for c in corpus]
        _BM25 = BM25Okapi(tokenized)
    return _DF, _BM25

def bm25_search(query: str, k=8, filename=None, category=None):
    """
    BM25 retrieval with optional filename/category filters.
    Returns a DataFrame slice retaining original indices.
    """
    df, bm25 = _init()
    scores = bm25.get_scores(_tokenize(query))
    out = df.join(pd.Series(scores, index=df.index, name="bm25"))

    if filename:
        fn = str(filename).lower()
        out = out[out["filename"].str.lower().str.contains(fn, na=False)]
    if category:
        out = out[out["category"].str.lower() == str(category).lower()]

    out = out.sort_values("bm25", ascending=False).head(k)
    return out
