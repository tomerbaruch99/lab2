"""
Answer Confidence Meter
=======================

Calculates confidence scores for RAG answers based on:
1. Average similarity between query and retrieved chunks (50%)
2. Retrieval overlap score - how much chunks agree with each other (30%)
3. Supported claim ratio - whether answer contains unsupported claims (20%)
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from utils import (
    CONFIDENCE_WEIGHTS,
    CONFIDENCE_THRESHOLDS,
    SIMILARITY_THRESHOLD_HIGH,
    SIMILARITY_THRESHOLD_MEDIUM,
    SUPPORTED_CLAIM_THRESHOLD,
    SUPPORTED_CLAIM_RATIO_HIGH,
    SUPPORTED_CLAIM_RATIO_MEDIUM,
)

try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    # Fallback cosine similarity function
    def cosine_similarity(X, Y=None):
        """Simple cosine similarity fallback."""
        if Y is None:
            Y = X
        X = np.array(X)
        Y = np.array(Y)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
        return np.dot(X_norm, Y_norm.T)


def calculate_avg_chunk_similarity(chunks: List[Dict]) -> float:
    """
    Calculate average similarity score between query and retrieved chunks.
    
    Args:
        chunks: List of chunk dictionaries with 'score' field
        
    Returns:
        Average similarity score (0-1)
    """
    if not chunks:
        return 0.0
    
    scores = [chunk.get("score", 0.0) for chunk in chunks]
    # Pinecone scores are typically 0-1 for cosine similarity
    # If they're in a different range, normalize them
    avg_score = np.mean(scores)
    
    # Normalize to 0-1 range if needed (assuming scores might be -1 to 1)
    if avg_score < 0:
        avg_score = (avg_score + 1) / 2
    
    return min(max(avg_score, 0.0), 1.0)


def calculate_retrieval_overlap_score(
    chunks: List[Dict],
    embedding_model=None
) -> float:
    """
    Calculate how much the retrieved chunks agree with each other.
    Uses semantic similarity between chunk pairs.
    
    Args:
        chunks: List of chunk dictionaries with 'chunk_text_only' field
        embedding_model: Optional embedding model for semantic similarity
        
    Returns:
        Overlap score (0-1), where 1 means all chunks are very similar
    """
    if len(chunks) < 2:
        return 1.0  # Single chunk or no chunks = perfect agreement
    
    # If no embedding model provided, use a simple text-based approach
    if embedding_model is None:
        # Fallback: use simple keyword overlap
        texts = [chunk.get("chunk_text_only", "") for chunk in chunks]
        if not any(texts):
            return 0.5  # Default moderate score
        
        # Simple word overlap calculation
        word_sets = [set(text.lower().split()) for text in texts if text]
        if not word_sets:
            return 0.5
        
        # Calculate pairwise Jaccard similarity
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        return np.mean(similarities) if similarities else 0.5
    
    # Use embedding-based similarity
    try:
        texts = [chunk.get("chunk_text_only", "") for chunk in chunks]
        texts = [t for t in texts if t]  # Filter empty texts
        
        if len(texts) < 2:
            return 1.0
        
        # Embed all chunks
        embeddings = [embedding_model.embed_query(text) for text in texts]
        embeddings = np.array(embeddings)
        
        # Calculate pairwise cosine similarities
        if HAS_SKLEARN:
            similarity_matrix = cosine_similarity(embeddings)
        else:
            # Manual calculation
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / (norms + 1e-8)
            similarity_matrix = np.dot(normalized, normalized.T)
        
        # Get upper triangle (excluding diagonal)
        n = len(similarity_matrix)
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(similarity_matrix[i][j])
        
        return float(np.mean(similarities)) if similarities else 0.5
    
    except Exception as e:
        print(f"[WARN] Error calculating overlap with embeddings: {e}")
        return 0.5  # Fallback to moderate score


def detect_unsupported_claims(
    answer: str,
    chunks: List[Dict],
    embedding_model=None,
    threshold: float = None
) -> Tuple[float, List[str]]:
    """
    Detect if the answer contains information not found in retrieved chunks.
    
    Args:
        answer: Generated answer text
        chunks: Retrieved chunks
        embedding_model: Optional embedding model for semantic similarity
        threshold: Minimum similarity threshold to consider a claim supported
                   (defaults to SUPPORTED_CLAIM_THRESHOLD from config)
        
    Returns:
        Tuple of (supported_claim_ratio, unsupported_claims_list)
        supported_claim_ratio: 0-1, where 1 means all claims are supported
    """
    if threshold is None:
        threshold = SUPPORTED_CLAIM_THRESHOLD
    
    if not answer or not chunks:
        return 0.5, []
    
    # Extract chunk texts
    chunk_texts = [chunk.get("chunk_text_only", "") for chunk in chunks]
    chunk_texts = [t for t in chunk_texts if t]  # Filter empty
    
    if not chunk_texts:
        return 0.5, []
    
    # Simple approach: split answer into sentences and check each
    # For Hebrew text, use simple sentence splitting
    import re
    sentences = re.split(r'[.!?]\s+', answer)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 1.0, []
    
    if embedding_model is None:
        # Fallback: keyword-based matching
        answer_words = set(answer.lower().split())
        chunk_words = set()
        for text in chunk_texts:
            chunk_words.update(text.lower().split())
        
        # Check how many answer words appear in chunks
        supported_words = answer_words & chunk_words
        if len(answer_words) == 0:
            return 1.0, []
        
        ratio = len(supported_words) / len(answer_words)
        return ratio, []
    
    # Use embedding-based similarity
    try:
        # Embed answer sentences
        answer_embeddings = [embedding_model.embed_query(s) for s in sentences]
        
        # Embed all chunks (or combine them)
        combined_chunk_text = " ".join(chunk_texts)
        chunk_embedding = embedding_model.embed_query(combined_chunk_text)
        
        # Check similarity of each sentence to chunks
        supported_count = 0
        unsupported = []
        
        for i, sent_emb in enumerate(answer_embeddings):
            # Calculate similarity to combined chunk text
            if HAS_SKLEARN:
                similarity = cosine_similarity(
                    np.array([sent_emb]),
                    np.array([chunk_embedding])
                )[0][0]
            else:
                # Manual cosine similarity
                sent_norm = sent_emb / (np.linalg.norm(sent_emb) + 1e-8)
                chunk_norm = chunk_embedding / (np.linalg.norm(chunk_embedding) + 1e-8)
                similarity = np.dot(sent_norm, chunk_norm)
            
            if similarity >= threshold:
                supported_count += 1
            else:
                unsupported.append(sentences[i])
        
        ratio = supported_count / len(sentences) if sentences else 1.0
        return ratio, unsupported
    
    except Exception as e:
        print(f"[WARN] Error detecting unsupported claims: {e}")
        return 0.7, []  # Fallback to moderate score


def _generate_confidence_reason(
    avg_sim: float,
    overlap: float,
    supported_ratio: float
) -> str:
    """
    Generate human-readable explanation for confidence score.
    
    Args:
        avg_sim: Average chunk similarity score
        overlap: Retrieval overlap score
        supported_ratio: Supported claim ratio
        
    Returns:
        Reason text string
    """
    reasons = []
    
    # Similarity reason
    if avg_sim >= SIMILARITY_THRESHOLD_HIGH:
        reasons.append("high similarity between user query and retrieved chunks")
    elif avg_sim >= SIMILARITY_THRESHOLD_MEDIUM:
        reasons.append("moderate similarity between user query and retrieved chunks")
    else:
        reasons.append("low similarity between user query and retrieved chunks")
    
    # Overlap reason
    if overlap >= SIMILARITY_THRESHOLD_HIGH:
        reasons.append("retrieved chunks strongly agree with each other")
    elif overlap >= SIMILARITY_THRESHOLD_MEDIUM:
        reasons.append("retrieved chunks moderately agree with each other")
    else:
        reasons.append("retrieved chunks show some disagreement")
    
    # Supported claims reason
    if supported_ratio >= SUPPORTED_CLAIM_RATIO_HIGH:
        reasons.append("no unsupported claims detected")
    elif supported_ratio >= SUPPORTED_CLAIM_RATIO_MEDIUM:
        reasons.append("most claims appear to be supported")
    else:
        reasons.append("some claims may not be fully supported by retrieved chunks")
    
    return "; ".join(reasons) + "."


def calculate_confidence(
    chunks: List[Dict],
    answer: str,
    embedding_model=None
) -> Dict[str, Any]:
    """
    Calculate overall confidence score for a RAG answer.
    
    Args:
        chunks: Retrieved chunks with 'score' and 'chunk_text_only' fields
        answer: Generated answer text
        embedding_model: Optional embedding model for advanced calculations
        
    Returns:
        Dictionary with:
        - confidence_score: Overall score (0-100)
        - confidence_level: "High", "Medium", or "Low"
        - avg_chunk_similarity: Average query-chunk similarity
        - retrieval_overlap: How much chunks agree
        - supported_claim_ratio: Ratio of supported claims
        - reason: Human-readable explanation
        - unsupported_claims: List of potentially unsupported claims
    """
    # Calculate components
    avg_sim = calculate_avg_chunk_similarity(chunks)
    overlap = calculate_retrieval_overlap_score(chunks, embedding_model)
    supported_ratio, unsupported = detect_unsupported_claims(
        answer, chunks, embedding_model
    )
    
    # Weighted combination using config weights
    weights = CONFIDENCE_WEIGHTS
    confidence_score = (
        weights["avg_chunk_similarity"] * avg_sim +
        weights["retrieval_overlap"] * overlap +
        weights["supported_claim_ratio"] * supported_ratio
    ) * 100  # Convert to percentage
    
    # Determine confidence level using config thresholds
    thresholds = CONFIDENCE_THRESHOLDS
    if confidence_score >= thresholds["high"]:
        level = "High"
    elif confidence_score >= thresholds["medium"]:
        level = "Medium"
    else:
        level = "Low"
    
    # Generate reason using config thresholds
    reason_text = _generate_confidence_reason(avg_sim, overlap, supported_ratio)
    
    return {
        "confidence_score": round(confidence_score, 1),
        "confidence_level": level,
        "avg_chunk_similarity": round(avg_sim * 100, 1),
        "retrieval_overlap": round(overlap * 100, 1),
        "supported_claim_ratio": round(supported_ratio * 100, 1),
        "reason": reason_text,
        "unsupported_claims": unsupported[:3],  # Limit to first 3
    }

