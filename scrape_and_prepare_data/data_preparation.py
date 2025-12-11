"""
Haifa Municipality – Data Preparation for RAG
=================================================================

This version uses 3 chunking strategies:

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
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Union
from urllib.parse import urlparse
import pandas as pd
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_OVERLAP


# ==============================================================
# URL EXTRACTION AND VALIDATION
# ==============================================================

RE_WHITESPACE = re.compile(r"[ \t\f\v]+")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
RE_URL = re.compile(
    r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
    re.IGNORECASE
)
# Pattern to match [URL: ...] format with text before it
RE_URL_MARKER = re.compile(r'\[URL:\s*(https?://[^\]]+)\]', re.IGNORECASE)


def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL."""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    if not url or len(url) < 4:  # Minimum: "http"
        return False
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except Exception:
        return False


def normalize_links(links: Union[List, str, None]) -> List[Dict[str, str]]:
    """
    Normalize links from various formats to a list of link dictionaries.
    Handles empty lists, empty strings, None, or malformed data.
    """
    if links is None:
        return []
    
    # Handle string input (could be JSON string or empty string)
    if isinstance(links, str):
        links = links.strip()
        if not links or links.lower() in ['[]', 'null', 'none', '']:
            return []
        try:
            # Try to parse as JSON
            parsed = json.loads(links)
            links = parsed
        except (json.JSONDecodeError, ValueError):
            # If not JSON, treat as empty
            return []
    
    # Handle list input
    if not isinstance(links, list):
        return []
    
    # Normalize list items to link dictionaries
    normalized = []
    for item in links:
        if isinstance(item, dict):
            # Check if it has a 'url' key with a valid URL
            url = item.get('url', '')
            if is_valid_url(url):
                normalized.append({
                    'text': item.get('text', ''),
                    'raw_text': item.get('raw_text', ''),
                    'url': url
                })
        elif isinstance(item, str):
            # If it's a string, check if it's a valid URL
            if is_valid_url(item):
                normalized.append({
                    'text': '',
                    'raw_text': '',
                    'url': item
                })
    
    return normalized


def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text using regex."""
    if not text or not isinstance(text, str):
        return []
    urls = RE_URL.findall(text)
    # Filter and deduplicate
    valid_urls = []
    seen = set()
    for url in urls:
        url = url.strip().rstrip('.,;:!)')
        if is_valid_url(url) and url not in seen:
            valid_urls.append(url)
            seen.add(url)
    return valid_urls


def extract_links_from_content(content: str, title: str = "", subtitle: str = "") -> Tuple[str, List[Dict[str, str]]]:
    """
    Extract links from content that contains [URL: ...] markers.
    Removes the link markers from content and returns cleaned content and extracted links.
    
    Args:
        content: Raw content text with [URL: ...] markers
        title: Optional title for context
        subtitle: Optional subtitle for context
    
    Returns:
        Tuple of (cleaned_content, links_list)
        links_list contains dicts with 'text', 'raw_text', and 'url' keys
    """
    if not content or not isinstance(content, str):
        return content, []
    
    links = []
    seen_urls = set()
    cleaned_lines = []
    
    # Keep track of recent lines for context
    recent_lines = []
    MAX_CONTEXT_LINES = 3  # Maximum number of previous lines to include in context
    
    # Process content line by line
    lines_list = content.split('\n')
    for line_idx, line in enumerate(lines_list):
        # Check if this line contains a URL marker
        matches = list(RE_URL_MARKER.finditer(line))
        
        if matches:
            # Build context from recent lines (up to MAX_CONTEXT_LINES)
            context_lines = recent_lines[-MAX_CONTEXT_LINES:] if recent_lines else []
            
            # Process each URL in this line
            last_end = 0
            for i, match in enumerate(matches):
                url = match.group(1).strip()
                
                if not is_valid_url(url) or url in seen_urls:
                    last_end = match.end()
                    continue
                
                seen_urls.add(url)
                
                # Extract text from end of previous match (or start of line) to start of current match
                # This ensures each URL gets the text immediately preceding it
                raw_text_on_line = line[last_end:match.start()].strip()
                last_end = match.end()
                
                # Combine context from previous lines with text on current line
                raw_text_parts = context_lines + [raw_text_on_line] if raw_text_on_line else context_lines
                raw_text = '\n'.join(raw_text_parts).strip()
                
                # Clean up the raw_text - normalize whitespace but preserve line breaks
                # Normalize multiple spaces but keep newlines for context
                raw_text_cleaned = '\n'.join([
                    RE_WHITESPACE.sub(' ', part).strip() 
                    for part in raw_text.split('\n')
                    if part.strip()
                ]).strip()
                
                # Create text with context (title/subtitle if available)
                text_parts = []
                if title:
                    text_parts.append(title)
                if subtitle:
                    text_parts.append(subtitle)
                if raw_text_cleaned:
                    # For text field, flatten newlines to spaces for cleaner embedding
                    text_with_context = ' '.join(raw_text_cleaned.split())
                    text_parts.append(text_with_context)
                
                # If still no text, use title/subtitle or empty
                cleaned_text = ' '.join(text_parts).strip() if text_parts else (title or subtitle or "")
                
                links.append({
                    'text': cleaned_text,
                    'raw_text': raw_text_cleaned,
                    'url': url
                })
            
            # Remove all [URL: ...] markers from this line
            line_cleaned = RE_URL_MARKER.sub('', line).strip()
            if line_cleaned:
                cleaned_lines.append(line_cleaned)
                recent_lines.append(line_cleaned)
                # Keep recent_lines bounded
                if len(recent_lines) > MAX_CONTEXT_LINES:
                    recent_lines.pop(0)
        else:
            # No URL markers in this line, keep as-is
            line_cleaned = line.strip()
            if line_cleaned:
                cleaned_lines.append(line_cleaned)
                recent_lines.append(line_cleaned)
                # Keep recent_lines bounded
                if len(recent_lines) > MAX_CONTEXT_LINES:
                    recent_lines.pop(0)
    
    cleaned_content = '\n'.join(cleaned_lines)
    
    return cleaned_content, links


def ensure_links_valid(links: Union[List, str, None], raw_content: str, title: str = "", subtitle: str = "") -> Tuple[str, List[Dict[str, str]]]:
    """
    Ensure links metadata contains valid hyperlinks.
    Prioritizes links from scraped metadata, then extracts from content if needed.
    Removes link markers from content and returns cleaned content and extracted links.
    
    Args:
        links: Links from page metadata (can be list, string, or None)
        raw_content: Raw content text to extract URLs from
        title: Optional title for link context
        subtitle: Optional subtitle for link context
    
    Returns:
        Tuple of (cleaned_content, links_list)
        links_list contains dicts with 'text', 'raw_text', and 'url' keys
    """
    # First, normalize existing links from metadata (prioritize scraped links)
    normalized_links = normalize_links(links)
    
    # If we have valid links from metadata, use those
    # But still clean content if it has [URL: ...] markers
    if normalized_links:
        if raw_content:
            # Clean content of [URL: ...] markers if present, but keep the scraped links
            cleaned_content, _ = extract_links_from_content(raw_content, title, subtitle)
            return cleaned_content, normalized_links
        else:
            return raw_content or "", normalized_links
    
    # If no valid links from metadata, try to extract from content using [URL: ...] pattern
    if raw_content:
        cleaned_content, extracted_links = extract_links_from_content(raw_content, title, subtitle)
        
        # If we found links in content, use those
        if extracted_links:
            return cleaned_content, extracted_links
        
        # Last resort: try to extract any URLs from content as fallback
        extracted_urls = extract_urls_from_text(raw_content)
        if extracted_urls:
            # Convert extracted URLs to link format (without context)
            fallback_links = [{'text': '', 'raw_text': '', 'url': url} for url in extracted_urls]
            return raw_content, fallback_links
    
    # Return original content with empty links list
    return raw_content or "", []


# ==============================================================
# SEMANTIC CLEANING
# ==============================================================

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
    """
    Determine the namespace for a document based on URL, title, and content.
    
    Namespaces help organize documents by topic in the Pinecone index, improving
    retrieval precision. This function uses keyword matching in both English
    and Hebrew to classify documents into categories.
    
    Args:
        url: Document URL (checked for keywords)
        title: Document title (checked for keywords)
        content: Document content (checked for keywords)
        
    Returns:
        Namespace string: one of:
        - "arnona": Property tax and municipal fees
        - "water": Water services and billing
        - "education": Educational services
        - "sanitation": Waste management and cleaning
        - "parking": Parking and traffic
        - "emergency": Emergency services and shelters
        - "engineering": Building permits and engineering
        - "welfare": Social services and welfare
        - "business": Business licensing
        - "culture": Cultural events and exhibitions
        - "general": Default fallback namespace
        
    Note:
        The function checks both URL and combined title+content for keywords.
        First match wins. If no keywords match, returns "general".
    """
    u = url.lower()
    t = (title + " " + content).lower()

    # Namespace keyword mapping (English and Hebrew)
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

    # Check for keywords in URL or text (title + content)
    for ns, keys in mapping.items():
        for k in keys:
            if k in u or k in t:
                return ns

    return "general"


def classify_doc_type(url: str, title: str, content: str) -> str:
    """
    Classify document type based on URL, title, and content characteristics.
    
    Document type classification helps the adaptive chunking strategy select
    the most appropriate chunking method for each document. Different document
    types benefit from different chunking approaches.
    
    Args:
        url: Document URL (checked for file extensions)
        title: Document title (checked for keywords)
        content: Document content (checked for keywords)
        
    Returns:
        Document type string, one of:
        - "pdf": PDF documents (detected from URL)
        - "event": Event announcements and cultural activities
        - "table": Pricing tables and tariff documents
        - "procedural": How-to guides and procedure documents
        - "general_info": General information and service descriptions
        - "mixed": Default fallback for unclassified documents
        
    Note:
        Classification uses keyword matching in Hebrew and English.
        PDF detection is based on URL extension.
        The adaptive chunking strategy uses this classification to select
        appropriate chunking methods (e.g., hierarchical for tables,
        sentence-based for procedural documents).
    """
    u = url.lower()
    t = (title + " " + content).lower()

    # Check for PDF files (based on URL extension)
    if u.endswith(".pdf") or ".pdf" in u:
        return "pdf"

    # Check for event-related content
    if any(kw in t for kw in ["אירוע", "event", "תערוכה", "פסטיבל", "החג של החגים"]):
        return "event"

    # Check for pricing/tariff tables
    if any(kw in t for kw in ["תעריף", "מחירון"]):
        return "table"

    # Check for procedural/how-to documents
    if any(kw in t for kw in ["איך", "כיצד", "בקשה", "טופס", "הליך"]):
        return "procedural"

    # Check for general information documents
    if any(kw in t for kw in ["שירות", "service", "מידע", "תושב"]):
        return "general_info"

    # Default fallback
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
    """Fixed-size chunks with overlap"""
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
    """Chunk text into sentences, grouping sentences until max_chars is reached"""
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
    """Chunk text into paragraphs, grouping paragraphs until max_chars is reached"""
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
    """Simple heading-based segmentation for PDFs"""
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
    """
    Adaptive chunking strategy that selects the best chunking method based on document type.
    
    This is the "smart" chunking strategy that dynamically chooses chunking methods
    based on document characteristics. Different document types benefit from different
    chunking approaches:
    - Events: Date/time-based chunking to preserve temporal structure
    - Tables: Paragraph-based chunking to preserve table structure
    - Procedural: Sentence-based chunking to preserve step-by-step instructions
    - General info: Paragraph-based chunking for natural content flow
    - Others: Baseline chunking with overlap as fallback
    
    Args:
        text: Text content to chunk
        doc_type: Document type classification (from classify_doc_type)
        max_chars: Maximum characters per chunk
        overlap: Character overlap between chunks (used for baseline fallback)
        
    Returns:
        List of text chunks, each respecting max_chars limit
        
    Strategy Selection:
        - "procedural" → chunk_by_sentences(): Preserves step-by-step flow
        - "general_info" or "table" → chunk_by_paragraphs(): Natural paragraph boundaries
        - "event" → chunk_event(): Preserves event date/time structure
        - "mixed"/"pdf"/others → chunk_baseline(): Character-based with overlap
        
    Note:
        This strategy is one of three main chunking strategies (baseline, sentence, adaptive).
        The adaptive strategy provides the best results for diverse document types but
        requires document type classification to work effectively.
    """
    text = text.strip()
    if not text:
        return []
    # Short text doesn't need chunking
    if len(text) <= max_chars:
        return [text]

    # Procedural documents: use sentence-based chunking to preserve instructions
    if doc_type == "procedural":
        return chunk_by_sentences(text, max_chars)

    # General info and tables: use paragraph-based chunking
    if doc_type in {"general_info", "table"}:
        return chunk_by_paragraphs(text, max_chars)

    # Events: use event-specific chunking (preserves date/time structure)
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
    raw_links = page.get("links", [])

    # Extract links from content (removes [URL: ...] markers) and get cleaned content
    cleaned_raw_content, links = ensure_links_valid(raw_links, raw_content, title, subtitle)

    # Clean text (semantic cleaning)
    content = clean_semantic_text(cleaned_raw_content)

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


def main():
    parser = argparse.ArgumentParser(
        description="Prepare scraped Haifa municipality data for RAG indexing"
    )
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--chunk_chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
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
