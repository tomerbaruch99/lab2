"""
Query Understanding Module
==========================
Maps free-form user questions to CUAD categories using zero-shot classification.
"""

from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import warnings

# CUAD categories from data_preparation.py
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

# Question templates for better matching
QUESTION_TEMPLATES = {
    "Document Name": "What is the name of this agreement?",
    "Parties": "Who are the parties to this agreement?",
    "Agreement Date": "On what date was the agreement executed?",
    "Effective Date": "When does the agreement become effective?",
    "Expiration Date": "When does the agreement's initial term expire?",
    "Renewal Term": "What is the renewal term after the initial term expires?",
    "Notice Period To Terminate Renewal": "What notice is required to terminate the renewal?",
    "Governing Law": "Which jurisdiction's law governs this agreement?",
    "Most Favored Nation": "Is there a most-favored-nation clause?",
    "Non-Compete": "Is there a non-compete restriction?",
    "Exclusivity": "Is there an exclusivity obligation?",
    "No-Solicit Of Customers": "Is there a restriction on soliciting customers?",
    "Competitive Restriction Exception": "Are there exceptions to competitive restrictions?",
    "No-Solicit Of Employees": "Is there a restriction on soliciting or hiring employees?",
    "Non-Disparagement": "Is there a non-disparagement requirement?",
    "Termination For Convenience": "Can the agreement be terminated without cause?",
    "Rofr/Rofo/Rofn": "Is there a right of first refusal/offer/negotiation?",
    "Change Of Control": "Are there provisions triggered by a change of control?",
    "Anti-Assignment": "Is consent or notice required to assign the agreement?",
    "Revenue/Profit Sharing": "Is there revenue or profit sharing?",
    "Price Restrictions": "Are there restrictions on pricing changes?",
    "Minimum Commitment": "Is there a minimum purchase or commitment?",
    "Volume Restriction": "Are there volume thresholds with constraints or fees?",
    "Ip Ownership Assignment": "Is IP assigned to the counterparty under any conditions?",
    "Joint Ip Ownership": "Is any IP jointly owned?",
    "License Grant": "Does the agreement grant a license?",
    "Non-Transferable License": "Is the license non-transferable?",
    "Affiliate License-Licensor": "Does the license include the licensor's affiliates or their IP?",
    "Affiliate License-Licensee": "Does the license extend to the licensee's affiliates?",
    "Unlimited/All-You-Can-Eat-License": "Is there an unlimited or enterprise-wide license?",
    "Irrevocable Or Perpetual License": "Is the license irrevocable or perpetual?",
    "Source Code Escrow": "Is source code escrow required?",
    "Post-Termination Services": "Are there post-termination service obligations?",
    "Audit Rights": "Are there audit rights?",
    "Uncapped Liability": "Is liability uncapped for any breach?",
    "Cap On Liability": "Is there a cap on liability?",
    "Liquidated Damages": "Are there liquidated damages or termination fees?",
    "Warranty Duration": "What is the duration of the warranty?",
    "Insurance": "Is insurance required?",
    "Covenant Not To Sue": "Is there a covenant not to sue?",
    "Third Party Beneficiary": "Are there third-party beneficiaries?",
}

# Keyword mappings for common queries
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

# Initialize model (lazy loading)
_model = None

def _get_model():
    """Lazy load the embedding model."""
    global _model
    if _model is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _model

def map_query_to_category(query: str, confidence_threshold: float = 0.3) -> Tuple[Optional[str], float]:
    """
    Map a user query to a CUAD category using zero-shot classification.
    
    Args:
        query: User's question
        confidence_threshold: Minimum confidence to return a category (default: 0.3)
    
    Returns:
        Tuple of (category_name, confidence_score) or (None, 0.0) if low confidence
    """
    query_lower = query.lower()
    
    # First, check keyword mappings (fast path)
    for keyword, category in KEYWORD_MAPPINGS.items():
        if keyword in query_lower:
            return category, 0.8  # High confidence for keyword match
    
    # Zero-shot classification using semantic similarity
    model = _get_model()
    
    # Encode query
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    
    # Encode all category question templates
    category_texts = [f"{cat}: {QUESTION_TEMPLATES.get(cat, cat)}" for cat in CUAD_CATEGORIES]
    category_embeddings = model.encode(category_texts, convert_to_numpy=True)
    
    # Compute cosine similarity
    similarities = np.dot(category_embeddings, query_embedding) / (
        np.linalg.norm(category_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get best match
    best_idx = np.argmax(similarities)
    best_score = float(similarities[best_idx])
    
    if best_score >= confidence_threshold:
        return CUAD_CATEGORIES[best_idx], best_score
    else:
        return None, best_score

def get_category_candidates(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Get top-k category candidates for a query.
    
    Args:
        query: User's question
        top_k: Number of candidates to return
    
    Returns:
        List of (category_name, confidence_score) tuples
    """
    model = _get_model()
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    
    category_texts = [f"{cat}: {QUESTION_TEMPLATES.get(cat, cat)}" for cat in CUAD_CATEGORIES]
    category_embeddings = model.encode(category_texts, convert_to_numpy=True)
    
    similarities = np.dot(category_embeddings, query_embedding) / (
        np.linalg.norm(category_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get top-k
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(CUAD_CATEGORIES[idx], float(similarities[idx])) for idx in top_indices]

