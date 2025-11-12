import re
from typing import List, Dict, Any

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def exactish_match(pred: str, ref: str) -> float:
    p, r = normalize(pred), normalize(ref)
    if r in p or p in r:
        return 1.0
    return 0.0

def compute_retrieval_metrics(used_docs: List[Dict[str,Any]], ground_filename: str, ground_category: str):
    """
    Success@k if any used chunk matches filename or category.
    """
    ok_file = any(d.get("filename") == ground_filename for d in used_docs)
    ok_cat  = any(d.get("category") == ground_category for d in used_docs)
    return {"hit_filename": int(ok_file), "hit_category": int(ok_cat)}
