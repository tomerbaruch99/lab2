"""
Haifa Municipality – Data Preparation for RAG
=================================================================

This version uses ONLY 3 chunking strategies:

    1. baseline
    2. sentence
    3. adaptive  (dynamic choice among 5 internal strategies)

Adaptive internally selects:
    - sentence
    - paragraph
    - event
    - hierarchical
    - baseline fallback

Based on:
    - doc_type  (pdf/event/procedural/general_info/mixed)
    - content length

This file outputs ONE unified parquet/csv containing:
    • text
    • chunking_strategy
    • doc_type
    • namespace
    • links
    • metadata
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm


# ==============================================================
# SEMANTIC CLEANING
# ==============================================================

RE_WHITESPACE = re.compile(r"[ \t\f\v]+")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")

MEANINGLESS_PHRASES = {
    "לחץ כאן", "למידע נוסף", "קראו עוד", "קרא עוד",
    "לפרטים", "פרטים נוספים", "המשך", "לצפייה"
}


def clean_semantic_text(text: str) -> str:
    """Removes UI noise, breadcrumbs, excessive whitespace."""
    if not text or pd.isna(text):
        return ""

    text = RE_WHITESPACE.sub(" ", str(text))
    text = RE_MULTI_NEWLINE.sub("\n\n", text).strip()

    for phrase in MEANINGLESS_PHRASES:
        text = text.replace(phrase, "")

    # Remove breadcrumbs like "בית > שירות לתושבים"
    cleaned = []
    for line in text.split("\n"):
        if ">" in line and line.count(">") <= 4:
            continue
        cleaned.append(line.strip())

    return "\n".join(l for l in cleaned if l).strip()


# ==============================================================
# DOCUMENT-TYPE CLASSIFIER + NAMESPACE
# ==============================================================

def compute_namespace(url: str, title: str, content: str) -> str:
    u = url.lower()
    t = (title + " " + content).lower()

    mapping = {
        "arnona": ["arnona", "ארנונה"],
        "water": ["water", "מים"],
        "education": ["education", "חינוך"],
        "sanitation": ["sanitation", "תברואה", "ניקיון", "אשפה"],
        "parking": ["parking", "חניה"],
        "emergency": ["emergency", "חירום", "מטה חירום", "מקלט", "אזעקה"],
        "engineering": ["engineering", "הנדסה"],
        "welfare": ["welfare", "רווחה"],
        "business": ["business", "עסקים", "רישוי"],
        "culture": ["culture", "אירועים", "מופע", "תערוכה"],
    }

    for ns, keys in mapping.items():
        for k in keys:
            if k in u or k in t:
                return ns

    return "general"


def classify_doc_type(url: str, title: str, content: str) -> str:
    u = url.lower()
    t = (title + " " + content).lower()

    if u.endswith(".pdf") or ".pdf" in u:
        return "pdf"

    if any(kw in t for kw in ["אירוע", "event", "תערוכה", "פסטיבל", "החג של החגים"]):
        return "event"

    if any(kw in t for kw in ["תעריף", "מחירון"]):
        return "table"

    if any(kw in t for kw in ["איך", "כיצד", "בקשה", "טופס", "הליך"]):
        return "procedural"

    if any(kw in t for kw in ["שירות", "service", "מידע", "תושב"]):
        return "general_info"

    return "mixed"


# ==============================================================
# CHUNKING HELPERS
# ==============================================================

def split_to_sentences(text: str) -> List[str]:
    """Naive sentence splitter."""
    if not text:
        return []
    normalized = text.replace("\n", " ")
    parts = re.split(r"([.!?])", normalized)
    sentences = []
    buf = ""
    for p in parts:
        if p in [".", "!", "?"]:
            buf += p
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
        else:
            buf += p
    if buf.strip():
        sentences.append(buf.strip())
    return [s for s in sentences if s]


def split_to_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


# ==============================================================
# THE THREE OFFICIAL STRATEGIES
# ==============================================================

# 1. BASELINE --------------------------------------------------
def chunk_baseline(text: str, max_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(start + max_chars - overlap, end)  # prevent infinite loop
    return chunks


# 2. SENTENCE --------------------------------------------------
def chunk_by_sentences(text: str, max_chars: int) -> List[str]:
    sents = split_to_sentences(text)
    if not sents:
        return []
    chunks = []
    buf = ""
    for s in sents:
        if len(buf) + len(s) + 1 <= max_chars:
            buf += (" " if buf else "") + s
        else:
            if buf.strip():
                chunks.append(buf.strip())
            buf = s
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


# INTERNAL SUB-STRATEGIES FOR ADAPTIVE -------------------------
def chunk_by_paragraphs(text: str, max_chars: int) -> List[str]:
    paras = split_to_paragraphs(text)
    if not paras:
        return []
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf += (("\n\n" if buf else "") + p)
        else:
            if buf.strip():
                chunks.append(buf.strip())
            buf = p
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def chunk_event(text: str, max_chars: int) -> List[str]:
    # For now same as paragraphs, but marked differently via strategy
    return chunk_by_paragraphs(text, max_chars)


def chunk_hierarchical(text: str, max_chars: int) -> List[str]:
    """Simple heading-based segmentation for PDFs."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return []
    chunks = []
    buf = ""
    for ln in lines:
        is_head = (
            len(ln) < 60 and
            (ln.endswith(":") or ln.isupper() or re.match(r"^\d+(\.\d+)*", ln))
        )
        if is_head and buf:
            chunks.append(buf.strip())
            buf = ln
        else:
            candidate = (buf + "\n" + ln) if buf else ln
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                chunks.append(buf.strip())
                buf = ln
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


# 3. ADAPTIVE --------------------------------------------------
def chunk_adaptive(text: str, doc_type: str, max_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    if doc_type == "procedural":
        return chunk_by_sentences(text, max_chars)

    if doc_type in {"general_info", "table"}:
        return chunk_by_paragraphs(text, max_chars)

    if doc_type == "event":
        return chunk_event(text, max_chars)

    if doc_type == "pdf":
        return chunk_hierarchical(text, max_chars)

    # fallback
    return chunk_baseline(text, max_chars, overlap)


# ==============================================================
# PROCESS SINGLE PAGE
# ==============================================================

def process_page(
    page: Dict,
    chunk_chars: int,
    chunk_overlap: int,
    strategies: List[str]
) -> List[Dict]:

    url = page.get("url", "") or ""
    title = (page.get("title", "") or "").strip()
    subtitle = (page.get("subtitle", "") or "").strip()
    raw_content = page.get("content", "") or ""
    links = page.get("links", []) or []

    # Clean text
    content = clean_semantic_text(raw_content)

    # Metadata
    doc_type = classify_doc_type(url, title, content)
    namespace = compute_namespace(url, title, content)
    doc_id = (
        url.replace("https://www.haifa.muni.il/", "")
           .replace("http://www.haifa.muni.il/", "")
           .replace("/", "_")
           .strip("_") or "homepage"
    )
    links_json = json.dumps(links, ensure_ascii=False)

    records = []

    for strategy in strategies:

        if strategy == "baseline":
            chunks = chunk_baseline(content, chunk_chars, chunk_overlap)

        elif strategy == "sentence":
            chunks = chunk_by_sentences(content, chunk_chars)

        elif strategy == "adaptive":
            chunks = chunk_adaptive(content, doc_type, chunk_chars, chunk_overlap)

        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        if not chunks:
            if content:
                chunks = [content[:chunk_chars]]
            else:
                continue

        for i, chunk_text in enumerate(chunks):
            parts = []
            if title:
                parts.append(f"כותרת: {title}")
            if subtitle:
                parts.append(f"תת-כותרת: {subtitle}")
            parts.append(chunk_text)
            full_text = "\n\n".join(parts)

            records.append({
                "doc_id": doc_id,
                "url": url,
                "title": title,
                "subtitle": subtitle,
                "doc_type": doc_type,
                "namespace": namespace,
                "chunk_id": i,
                "chunking_strategy": strategy,
                "text": full_text,
                "chunk_text_only": chunk_text,
                "links": links_json,
            })

    return records


# ==============================================================
# MAIN PIPELINE
# ==============================================================

def prepare_data(
    input_json: str,
    out_dir: str,
    chunk_chars: int,
    chunk_overlap: int,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    strategies = ["baseline", "sentence", "adaptive"]

    print(f"[LOAD] Scraped data: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks: List[Dict] = []

    print(f"[PROCESS] {len(pages)} pages → 3 chunking strategies...")
    for page in tqdm(pages, desc="Pages"):
        try:
            recs = process_page(page, chunk_chars, chunk_overlap, strategies)
            all_chunks.extend(recs)
        except Exception as e:
            print(f"[WARN] Failed on {page.get('url')}: {e}")

    df = pd.DataFrame(all_chunks)

    parquet_path = out_path / "haifa_rag_chunks.parquet"
    csv_path = out_path / "haifa_rag_chunks.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print("\n======= SUMMARY =======")
    print("Total pages:", len(pages))
    print("Total chunks:", len(df))
    print("\nChunks by strategy:")
    print(df["chunking_strategy"].value_counts())
    print("\nChunks by doc_type:")
    print(df["doc_type"].value_counts())
    print("\nChunks by namespace:")
    print(df["namespace"].value_counts())
    print("========================\n")

    return str(parquet_path)


# ==============================================================
# COMPATIBILITY FUNCTIONS (for old-legal_rag)
# ==============================================================

def clean_clause_text(text: str) -> str:
    """
    Compatibility function for old-legal_rag code.
    Cleans text similar to clean_semantic_text but simpler.
    """
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    if text.lower() in ("nan", "none", ""):
        return ""
    
    # Basic cleaning: normalize whitespace
    text = RE_WHITESPACE.sub(" ", text)
    text = RE_MULTI_NEWLINE.sub("\n\n", text)
    text = text.replace("\r", "\n").strip()
    
    return text


def chunk_text(text: str, max_chars: int = 4000, overlap: int = 400) -> List[Tuple[int, str]]:
    """
    Compatibility function for old-legal_rag code.
    Chunks text by characters with overlap, returns List[Tuple[int, str]].
    Similar to old chunk_text but uses new cleaning function.
    """
    if not text or pd.isna(text):
        return []
    
    text = clean_clause_text(str(text))
    if not text:
        return []
    
    chunks = []
    start = 0
    n = len(text)
    min_advance = max(1, max_chars - overlap)
    max_iterations = (n // min_advance) * 3 + 100
    iterations = 0
    
    while start < n:
        iterations += 1
        if iterations > max_iterations:
            print(f"[WARN] chunk_text: Max iterations ({max_iterations}) reached for text of length {n}, breaking at start={start}")
            break
        
        end = min(start + max_chars, n)
        # Try to end at a sentence boundary
        cut = text.rfind(". ", start, end)
        if cut == -1 or cut <= start + 200:
            cut = end
        else:
            cut += 2  # Include ". "
        
        chunk_text_slice = text[start:cut].strip()
        if chunk_text_slice:
            chunks.append((start, chunk_text_slice))
        
        if cut >= n:
            break
        
        next_start = max(start + min_advance, cut - overlap)
        if next_start >= n:
            break
        
        start = next_start
    
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--chunk_chars", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        raise FileNotFoundError(args.input_json)

    prepare_data(
        input_json=args.input_json,
        out_dir=args.out_dir,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
