"""
Haifa Municipality Data Preparation for RAG
===========================================
Prepares scraped Haifa municipality website data for RAG indexing.

What this script does:
1. Loads scraped JSON data from haifa_scraped.json
2. Cleans and processes the content
3. Chunks long content into smaller pieces with overlap
4. Creates structured format compatible with RAG indexing
5. Outputs parquet/CSV files for downstream indexing

Usage:
    python data_preparation.py \
        --input_json ./scrape_and_prepare_data/haifa_scraped.json \
        --out_dir ./scrape_and_prepare_data/haifa_prepared_data \
        --chunk_chars 1000 \
        --chunk_overlap 200
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
from tqdm import tqdm


# --- Text Processing Utilities ---

RE_WHITESPACE = re.compile(r"[ \t\f\v]+")
RE_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Clean text while preserving structure."""
    if not text or pd.isna(text):
        return ""
    text = str(text)
    # Normalize whitespace but keep newlines
    text = RE_WHITESPACE.sub(" ", text)
    # Reduce multiple newlines to double newlines
    text = RE_MULTIPLE_NEWLINES.sub("\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[Tuple[int, str]]:
    """
    Chunk text by characters with overlap, trying to break at sentence boundaries.
    
    Returns:
        List of (start_char, chunk_text) tuples
    """
    if not text:
        return []
    
    text = clean_text(text)
    if not text:
        return []
    
    chunks = []
    start = 0
    n = len(text)
    min_advance = max(1, max_chars - overlap)
    
    while start < n:
        end = min(start + max_chars, n)
        
        # Try to break at sentence boundary (period, exclamation, question mark)
        # Look for sentence endings within the last portion of the chunk (use overlap or 200, whichever is larger)
        lookback_window = max(overlap, 200)
        lookback_start = max(start, end - lookback_window)
        cut = end
        
        # Try to find sentence boundary
        for boundary in [". ", ".\n", "! ", "!\n", "? ", "?\n", ".\t", "!\t", "?\t"]:
            pos = text.rfind(boundary, lookback_start, end)
            if pos != -1 and pos > start + 100:  # Ensure minimum chunk size
                cut = pos + len(boundary)
                break
        
        # If no sentence boundary found, try paragraph boundary
        if cut == end:
            para_pos = text.rfind("\n\n", lookback_start, end)
            if para_pos != -1 and para_pos > start + 100:
                cut = para_pos + 2
        
        chunk_text_slice = text[start:cut].strip()
        if chunk_text_slice:
            chunks.append((start, chunk_text_slice))
        
        if cut >= n:
            break
        
        # Calculate next start with overlap
        next_start = max(start + min_advance, cut - overlap)
        if next_start >= n:
            break
        
        start = next_start
    
    return chunks


# --- Data Loading and Processing ---

def load_scraped_data(json_path: str) -> List[Dict]:
    """Load scraped data from JSON file."""
    print(f"[STEP] Loading scraped data from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} pages")
    return data


def process_page(page: Dict, chunk_chars: int, chunk_overlap: int) -> List[Dict]:
    """
    Process a single page into chunks.
    
    Returns:
        List of chunk dictionaries with metadata
    """
    url = page.get("url", "")
    title = page.get("title", "").strip()
    subtitle = page.get("subtitle", "").strip()
    content = page.get("content", "")
    
    # Create a document ID from URL
    doc_id = url.replace("https://www.haifa.muni.il/", "").replace("/", "_").strip("_") or "homepage"
    
    # Chunk the content
    chunks = chunk_text(content, max_chars=chunk_chars, overlap=chunk_overlap)
    
    if not chunks:
        # If no chunks, create one empty chunk to preserve the page
        chunks = [(0, content[:chunk_chars] if content else "")]
    
    # Create chunk records
    chunk_records = []
    for chunk_id, (start_char, chunk_text_content) in enumerate(chunks):
        # Build context-aware text with title/subtitle
        # Skip generic titles that don't provide meaningful context
        generic_titles = ["pdf document", "pdf", "document", "untitled", "untitled document"]
        is_generic_title = title.lower().strip() in generic_titles if title else False
        
        context_parts = []
        # Only include meaningful titles (skip generic ones like "PDF Document")
        if title and not is_generic_title:
            context_parts.append(f"כותרת: {title}")
        if subtitle:
            context_parts.append(f"תת-כותרת: {subtitle}")
        if chunk_text_content:
            context_parts.append(chunk_text_content)
        
        full_text = "\n\n".join(context_parts) if context_parts else chunk_text_content
        
        # Detect file type from URL
        file_type = "html"  # Default
        if url.lower().endswith(".pdf"):
            file_type = "pdf"
        elif url.lower().endswith((".doc", ".docx")):
            file_type = "doc"
        elif url.lower().endswith((".xls", ".xlsx")):
            file_type = "xls"
        elif url.lower().endswith((".txt", ".text")):
            file_type = "txt"
        elif ".pdf" in url.lower():
            file_type = "pdf"
        
        chunk_records.append({
            "doc_id": doc_id,
            "url": url,
            "filename": doc_id,  # For compatibility with indexing script
            "chunk_id": chunk_id,
            "start_char": start_char,
            "text": full_text,
            "title": title,
            "subtitle": subtitle,
            "chunk_text_only": chunk_text_content,
            "file_type": file_type,  # Add file type metadata
        })
    
    return chunk_records


def prepare_data(
    input_json: str,
    out_dir: str,
    chunk_chars: int = 1000,
    chunk_overlap: int = 200,
    config_suffix: str = None,
) -> str:
    """
    Main data preparation function.
    
    Args:
        input_json: Path to input JSON file
        out_dir: Output directory for prepared data
        chunk_chars: Maximum characters per chunk
        chunk_overlap: Character overlap between chunks
        config_suffix: Optional suffix for config-based filenames (e.g., "chunk1000_overlap200")
    
    Returns:
        Path to the created parquet file
    """
    # Create output directory
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate config suffix if not provided
    if config_suffix is None:
        config_suffix = f"chunk{chunk_chars}_overlap{chunk_overlap}"
    
    # Load data
    pages = load_scraped_data(input_json)
    
    # Process all pages into chunks
    print(f"[STEP] Processing {len(pages)} pages into chunks...")
    print(f"[INFO] Chunk size: {chunk_chars} chars, Overlap: {chunk_overlap} chars")
    print(f"[INFO] Config suffix: {config_suffix}")
    
    all_chunks = []
    for page in tqdm(pages, desc="Processing pages"):
        try:
            chunks = process_page(page, chunk_chars, chunk_overlap)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[WARN] Error processing page {page.get('url', 'unknown')}: {e}")
            continue
    
    if not all_chunks:
        raise ValueError("No chunks were created from the input data")
    
    # Create DataFrame
    print(f"[STEP] Creating DataFrame from {len(all_chunks)} chunks...")
    df = pd.DataFrame(all_chunks)
    
    # Save as Parquet (preferred for efficiency) with config suffix
    parquet_path = out_dir_path / f"haifa_paragraph_index_config_{config_suffix}.parquet"
    print(f"[STEP] Saving to {parquet_path}...")
    df.to_parquet(parquet_path, index=False)
    print(f"[OK] Saved {len(df):,} chunks to {parquet_path}")
    
    # Also save as CSV (fallback) with config suffix
    csv_path = out_dir_path / f"haifa_paragraph_index_config_{config_suffix}.csv"
    print(f"[STEP] Saving to {csv_path}...")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] Saved {len(df):,} chunks to {csv_path}")
    
    # Create a summary document index with config suffix
    doc_summary = df.groupby("doc_id").agg({
        "url": "first",
        "title": "first",
        "subtitle": "first",
        "chunk_id": "count",
    }).rename(columns={"chunk_id": "num_chunks"}).reset_index()
    
    summary_path = out_dir_path / f"haifa_document_index_config_{config_suffix}.parquet"
    doc_summary.to_parquet(summary_path, index=False)
    print(f"[OK] Saved document summary to {summary_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("[REPORT]")
    print("="*60)
    print(f"Config: {config_suffix}")
    print(f"Total pages processed: {len(pages)}")
    print(f"Total chunks created: {len(df):,}")
    print(f"Average chunks per page: {len(df) / len(pages):.1f}")
    print(f"Unique documents: {df['doc_id'].nunique()}")
    print(f"Average chunk length: {df['text'].str.len().mean():.0f} characters")
    print(f"Min chunk length: {df['text'].str.len().min()} characters")
    print(f"Max chunk length: {df['text'].str.len().max()} characters")
    print("="*60)
    
    return str(parquet_path)


def main():
    parser = argparse.ArgumentParser(
        description="Haifa Municipality Data Preparation for RAG"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="./scrape_and_prepare_data/haifa_scraped.json",
        help="Path to input JSON file with scraped data (default: ./scrape_and_prepare_data/haifa_scraped.json)"
    )
    parser.add_argument("--out_dir", type=str, default="./scrape_and_prepare_data/haifa_prepared_data",
                        help="Output directory for prepared data (default: ./scrape_and_prepare_data/haifa_prepared_data)")
    parser.add_argument("--chunk_chars", type=int, default=1000, help="Maximum characters per chunk")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Character overlap between chunks")
    parser.add_argument("--config_suffix", type=str, default=None,
                        help="Optional config suffix for filenames (default: auto-generated from chunk_chars and chunk_overlap)")
    parser.add_argument("--run_all_configs", action="store_true",
                        help="Run multiple chunk size/overlap configurations")
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_json):
        raise FileNotFoundError(f"Input file not found: {args.input_json}")
    
    if args.run_all_configs:
        # Run multiple configurations
        configs = [
            (500, 100),   # Small chunks, small overlap
            (750, 150),   # Medium-small chunks
            (1000, 200),  # Default
            (1500, 300),  # Larger chunks
            (2000, 400),  # Large chunks
        ]
        
        print(f"[INFO] Running {len(configs)} configurations...")
        for chunk_chars, chunk_overlap in configs:
            config_suffix = f"chunk{chunk_chars}_overlap{chunk_overlap}"
            print(f"\n{'='*60}")
            print(f"[CONFIG] chunk_chars={chunk_chars}, chunk_overlap={chunk_overlap}")
            print(f"{'='*60}\n")
            prepare_data(
                input_json=args.input_json,
                out_dir=args.out_dir,
                chunk_chars=chunk_chars,
                chunk_overlap=chunk_overlap,
                config_suffix=config_suffix,
            )
    else:
        # Run single configuration
        prepare_data(
            input_json=args.input_json,
            out_dir=args.out_dir,
            chunk_chars=args.chunk_chars,
            chunk_overlap=args.chunk_overlap,
            config_suffix=args.config_suffix,
        )


if __name__ == "__main__":
    main()

