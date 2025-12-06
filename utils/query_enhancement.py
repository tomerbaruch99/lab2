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


def rephrase_query(question: str, gemini_model) -> str:
    """
    Rephrase a question to be clearer and more precise without changing meaning.
    
    Args:
        question: Original user question
        gemini_model: Initialized Gemini model
        
    Returns:
        Rephrased question
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
    
    Args:
        question: Original user question
        gemini_model: Initialized Gemini model
        
    Returns:
        Enriched question with additional keywords
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
    
    Args:
        question: User question
        chunks: List of chunks from retriever
        top_k: Number of top chunks to return
        gemini_model: Initialized Gemini model
        
    Returns:
        Reranked list of top_k most relevant chunks
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
    indices = [int(i) for i in indices if int(i) >= 1 and int(i) <= len(chunks)]
    
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

