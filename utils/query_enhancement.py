"""
Query Enhancement and Reranking Utilities
=========================================

Advanced RAG features for improving query quality and retrieval:
- Query rephrasing
- Query enrichment
- Chunk reranking with LLM
"""

import json
import re
from typing import List, Dict, Optional

# Import call_gemini from gemini_integration to avoid duplication
import sys
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gemini_integration import call_gemini


def rephrase_query(question: str, gemini_model) -> str:
    """
    Rephrase a question to be clearer and more precise without changing meaning.
    
    This function uses an LLM (Gemini) to improve query clarity while preserving
    the original intent. Useful for handling ambiguous or poorly worded queries
    that might not retrieve optimal results.
    
    Args:
        question: Original user question in Hebrew
        gemini_model: Initialized Gemini GenerativeModel instance
        
    Returns:
        Rephrased question string, or original question if rephrasing fails
        
    Example:
        >>> rephrase_query("ארנונה איך?", gemini_model)
        'איך משלמים ארנונה?'
        
    Note:
        This function makes an API call to Gemini, so it has latency and cost.
        Use sparingly or cache results for common queries.
    """
    prompt = f"""
אתה עוזר AI. קבל שאלה של תושב חיפה וכתוב אותה מחדש בצורה ברורה ומדויקת יותר,
ללא שינוי המשמעות.

שאלה מקורית:
\"\"\"{question}\"\"\"

החזר רק את הניסוח המשופר, ללא הסברים נוספים.
"""
    resp = call_gemini(gemini_model, prompt)
    return resp.strip() or question


def enrich_query(question: str, gemini_model) -> str:
    """
    Enrich a question with additional keywords and hints for better retrieval.
    
    This function expands a query with related terms and context that improve
    semantic search results. For example, "ארנונה" might be enriched to
    "ארנונה תשלומים חיוב city4u" to help match more relevant documents.
    
    Args:
        question: Original user question in Hebrew
        gemini_model: Initialized Gemini GenerativeModel instance
        
    Returns:
        Enriched question string with additional keywords, or original if enrichment fails
        
    Example:
        >>> enrich_query("איך משלמים ארנונה?", gemini_model)
        'איך משלמים ארנונה תשלומים חיוב city4u?'
        
    Note:
        - Enrichment adds domain-specific terms (e.g., "city4u" for municipal services)
        - Does not change the core meaning of the question
        - Makes an API call to Gemini (has latency and cost)
        - Useful when retrieval results are suboptimal due to missing keywords
    """
    prompt = f"""
קבל שאלה של תושב חיפה על שירות עירוני. על בסיס השאלה,
הצע ניסוח מורחב שמוסיף מילות מפתח ורמזים חשובים לחיפוש במסמכים.

שאלה:
\"\"\"{question}\"\"\"

הנחיות:
1. אל תשנה את המשמעות.
2. הוסף מונחים נלווים (למשל: "ארנונה, תשלומים, חיוב, city4u").
3. החזר שורה אחת בלבד, עם השאלה המורחבת.

החזר רק את השאלה המורחבת, ללא הסברים.
"""
    resp = call_gemini(gemini_model, prompt)
    return resp.strip() or question


def rerank_chunks(question: str, chunks: List[Dict], top_k: int, gemini_model) -> List[Dict]:
    """
    Rerank retrieved chunks using LLM to select the most relevant ones.
    
    This function uses an LLM (Gemini) to intelligently rerank chunks based on
    semantic relevance to the query, rather than just similarity scores.
    This can improve answer quality by prioritizing chunks that are more
    contextually relevant, even if they have slightly lower similarity scores.
    
    Args:
        question: User question in Hebrew
        chunks: List of chunk dictionaries from retriever (should have more than top_k)
        top_k: Number of top chunks to return after reranking
        gemini_model: Initialized Gemini GenerativeModel instance
        
    Returns:
        Reranked list of top_k most relevant chunks, ordered by LLM-determined relevance
        
    Process:
        1. Formats chunks with metadata (title, URL, text preview) for LLM
        2. Asks LLM to select top_k most relevant chunks by number
        3. Parses LLM response to extract chunk indices
        4. Returns chunks in LLM-determined order
        
    Example:
        >>> chunks = retriever.retrieve("ארנונה", top_k=10)
        >>> reranked = rerank_chunks("ארנונה", chunks, top_k=5, gemini_model)
        >>> # Returns 5 chunks, ordered by LLM-determined relevance
        
    Note:
        - Requires more chunks than top_k (typically 2x) to be effective
        - Makes an API call to Gemini (has latency and cost)
        - Falls back to first top_k chunks if LLM response cannot be parsed
        - Useful when similarity scores don't perfectly reflect relevance
    """
    if not chunks:
        return []

    # Prepare a summary of each chunk for the prompt
    lines = []
    for i, ch in enumerate(chunks, 1):
        text = ch.get("chunk_text_only") or ch.get("text", "")
        text = str(text).replace("\n", " ")
        if len(text) > 350:
            text = text[:350] + "..."
        title = ch.get("title", "")
        url = ch.get("url", "")
        meta = []
        if title:
            meta.append(f"title: {title}")
        if url:
            meta.append(f"URL: {url}")
        meta_str = " | ".join(meta)
        lines.append(f"[{i}] {meta_str}\n{text}\n")

    chunks_str = "\n".join(lines)

    prompt = f"""
אתה מקבל שאלה של תושב חיפה ורשימת קטעי טקסט (מסומנים כמספרים [1], [2], ...).
עליך לבחור את הקטעים שהכי עוזרים לענות על השאלה.

שאלה:
\"\"\"{question}\"\"\"

קטעים:
{chunks_str}

הוראות:
1. בחר את {top_k} הקטעים הרלוונטיים ביותר.
2. החזר רק רשימה של המספרים, מופרדת בפסיקים, בסדר יורד של רלוונטיות.
3. לדוגמה: 3,1,5,2,4

החזר רק את הרשימה, ללא טקסט נוסף.
"""
    resp = call_gemini(gemini_model, prompt)
    text = resp.strip()
    
    # Try to extract numbers from the response
    indices = re.findall(r"\d+", text)
    # Convert to int and filter valid indices (1-based, so 1 to len(chunks))
    valid_indices = []
    for idx_str in indices:
        try:
            idx = int(idx_str)
            if 1 <= idx <= len(chunks):
                valid_indices.append(idx)
        except ValueError:
            continue
    indices = valid_indices
    
    # Remove duplicates and keep order
    seen = set()
    ordered = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    
    if not ordered:
        # If we didn't manage to parse the response, return the first ones
        return chunks[:top_k]
    
    # Map back to chunks (1-based -> 0-based)
    ordered_chunks = [chunks[i - 1] for i in ordered[:top_k]]
    return ordered_chunks

