"""
Query Understanding
Maps free-form user questions to CUAD categories (zero-shot-ish via similarity),
with a fast keyword path and question templates per category.
"""

from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import warnings

from data_preparation import QUESTION_TEMPLATES

CUAD_CATEGORIES = [
    "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date",
    "Renewal Term", "Notice Period To Terminate Renewal", "Governing Law",
    "Most Favored Nation", "Non-Compete", "Exclusivity", "No-Solicit Of Customers",
    "Competitive Restriction Exception", "No-Solicit Of Employees", "Non-Disparagement",
    "Termination For Convenience", "Rofr/Rofo/Rofn", "Change Of Control", "Anti-Assignment",
    "Revenue/Profit Sharing", "Price Restrictions", "Minimum Commitment", "Volume Restriction",
    "Ip Ownership Assignment", "Joint Ip Ownership", "License Grant", "Non-Transferable License",
    "Affiliate License-Licensor", "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License", "Source Code Escrow", "Post-Termination Services",
    "Audit Rights", "Uncapped Liability", "Cap On Liability", "Liquidated Damages",
    "Warranty Duration", "Insurance", "Covenant Not To Sue", "Third Party Beneficiary"
]

KEYWORD_MAPPINGS = {
    "termination": "Termination For Convenience",
    "terminate": "Termination For Convenience",
    "end without cause": "Termination For Convenience",
    "non-compete": "Non-Compete",
    "non compete": "Non-Compete",
    "exclusivity": "Exclusivity",
    "governing law": "Governing Law",
    "jurisdiction": "Governing Law",
    "assignment": "Anti-Assignment",
    "assign": "Anti-Assignment",
    "change of control": "Change Of Control",
    "liability cap": "Cap On Liability",
    "liability limit": "Cap On Liability",
    "uncapped liability": "Uncapped Liability",
    "insurance": "Insurance",
    "warranty": "Warranty Duration",
    "license": "License Grant",
    "ip ownership": "Ip Ownership Assignment",
    "parties": "Parties",
    "effective date": "Effective Date",
    "expiration": "Expiration Date",
    "renewal": "Renewal Term",
}

_model = None

def _get_model():
    """Lazy load a lightweight encoder for similarity."""
    global _model
    if _model is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _model

def map_query_to_category(query: str, confidence_threshold: float = 0.3) -> Tuple[Optional[str], float]:
    ql = (query or "").lower()

    for keyword, category in KEYWORD_MAPPINGS.items():
        if keyword in ql:
            return category, 0.8

    model = _get_model()
    qe = model.encode([query], convert_to_numpy=True)[0]
    cat_texts = [f"{c}: {QUESTION_TEMPLATES.get(c, c)}" for c in CUAD_CATEGORIES]
    ce = model.encode(cat_texts, convert_to_numpy=True)

    sims = np.dot(ce, qe) / (np.linalg.norm(ce, axis=1) * np.linalg.norm(qe) + 1e-12)
    idx = int(np.argmax(sims))
    score = float(sims[idx])

    if score >= confidence_threshold:
        return CUAD_CATEGORIES[idx], score
    return None, score

def get_category_candidates(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    model = _get_model()
    qe = model.encode([query], convert_to_numpy=True)[0]
    cat_texts = [f"{c}: {QUESTION_TEMPLATES.get(c, c)}" for c in CUAD_CATEGORIES]
    ce = model.encode(cat_texts, convert_to_numpy=True)
    sims = np.dot(ce, qe) / (np.linalg.norm(ce, axis=1) * np.linalg.norm(qe) + 1e-12)
    top = np.argsort(sims)[-top_k:][::-1]
    return [(CUAD_CATEGORIES[i], float(sims[i])) for i in top]
