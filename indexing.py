"""
Haifa Municipality Data Indexing
================================
Indexes prepared Haifa municipality data into Pinecone for RAG.

This script:
1. Loads prepared chunks from parquet/CSV
2. Generates embeddings using SentenceTransformer
3. Indexes vectors into Pinecone

Usage:
    python indexing.py \
        --prepared_dir ./scrape_and_prepare_data/haifa_prepared_data \
        --api_keys_path api_keys.json \
        --embedding_model intfloat/multilingual-e5-base \
        --index_name haifa-municipality-rag-index \
        --batch_size 128
"""

import os
import argparse
import hashlib
from typing import List, Dict, Iterable
from urllib.parse import quote

import pandas as pd
from tqdm import tqdm
from pinecone import Pinecone

from utils import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_NAME,
    DEFAULT_API_KEYS_PATH,
    load_pinecone_api_key,
    create_index,
    EmbeddingModel,
)


# --- Defaults (module-specific) ---

DEFAULT_PREPARED_DIR = "./scrape_and_prepare_data/haifa_prepared_data"
DEFAULT_PARAGRAPH_PARQUET = "haifa_paragraph_index_config_chunk1000_overlap200.parquet"
DEFAULT_PARAGRAPH_CSV = "haifa_paragraph_index_config_chunk1000_overlap200.csv"


# --- Data iterator ---

def iter_chunks(prepared_dir: str,
                paragraph_parquet: str,
                paragraph_csv: str,
                show_progress: bool = True) -> Iterable[Dict]:
    """
    Yield chunks from prepared data.
    Each yielded dict has:
      - text (full text with title/subtitle)
      - doc_id
      - filename
      - chunk_id
      - start_char
      - url (optional)
      - title (optional)
      - subtitle (optional)
      - chunk_text_only (optional, just the chunk content without title/subtitle)
      - file_type (optional, detected from URL: pdf, html, doc, xls, txt)
    """

    parquet_path = os.path.join(prepared_dir, paragraph_parquet)
    csv_path = os.path.join(prepared_dir, paragraph_csv)

    if os.path.exists(parquet_path):
        path = parquet_path
        ext = ".parquet"
    elif os.path.exists(csv_path):
        path = csv_path
        ext = ".csv"
    else:
        raise FileNotFoundError(
            f"No paragraph index found at {parquet_path} or {csv_path}"
        )

    print(f"[STEP] Loading chunks from {path} ...")

    if ext == ".parquet":
        df = pd.read_parquet(path)
        total_rows = len(df)
        print(f"[INFO] Found {total_rows:,} rows in parquet file")
        
        iterator = df.iterrows()
        if show_progress:
            iterator = tqdm(iterator, total=total_rows, desc="Reading chunks", unit="chunk")
        
        for _, row in iterator:
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            yield {
                "text": text,
                "doc_id": row.get("doc_id"),
                "filename": row.get("filename"),
                "chunk_id": int(row.get("chunk_id", 0)),
                "start_char": int(row.get("start_char", 0)),
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "subtitle": row.get("subtitle", ""),
                "chunk_text_only": row.get("chunk_text_only", ""),
                "file_type": row.get("file_type", "html"),
            }

    else:  # CSV
        if show_progress:
            print(f"[INFO] Counting rows in CSV file...")
            total_rows = sum(1 for _ in pd.read_csv(path, chunksize=50_000, usecols=[0]))
            print(f"[INFO] Found {total_rows:,} rows in CSV file")
            pbar = tqdm(total=total_rows, desc="Reading chunks", unit="chunk")
        else:
            pbar = None
        
        for chunk in pd.read_csv(path, chunksize=50_000):
            for _, row in chunk.iterrows():
                text = str(row.get("text") or "").strip()
                if not text:
                    if pbar:
                        pbar.update(1)
                    continue
                if pbar:
                    pbar.update(1)
                yield {
                    "text": text,
                    "doc_id": row.get("doc_id"),
                    "filename": row.get("filename"),
                    "chunk_id": int(row.get("chunk_id", 0)),
                    "start_char": int(row.get("start_char", 0)),
                    "url": row.get("url", ""),
                    "title": row.get("title", ""),
                    "subtitle": row.get("subtitle", ""),
                    "chunk_text_only": row.get("chunk_text_only", ""),
                    "file_type": row.get("file_type", "html"),
                }
        
        if pbar:
            pbar.close()


# --- Upsert helper ---

def sanitize_doc_id_for_vector_id(doc_id: str, max_length: int = 450) -> str:
    """
    Sanitize doc_id to be ASCII-safe for Pinecone vector IDs.
    
    Pinecone requires vector IDs to be:
    - ASCII-only
    - Maximum 512 characters total
    
    This function URL-encodes non-ASCII characters and truncates if needed
    to leave room for the chunk suffix (::chunk-{chunk_id}).
    
    Args:
        doc_id: Document ID (may contain non-ASCII characters)
        max_length: Maximum length to allow (default 450 to leave room for ::chunk-{id})
    
    Returns:
        ASCII-safe version of doc_id, truncated if necessary
    """
    if not doc_id:
        return ""
    # URL-encode non-ASCII characters to make it ASCII-safe
    # Use quote with safe='' to encode all special characters
    safe_doc_id = quote(str(doc_id), safe='')
    
    # Truncate if too long (leave room for ::chunk-{chunk_id} suffix)
    if len(safe_doc_id) > max_length:
        # Truncate and add indicator
        safe_doc_id = safe_doc_id[:max_length - 3] + "..."
    
    return safe_doc_id


def upsert_index(index,
                 embeddings: List[List[float]],
                 texts: List[str],
                 doc_ids: List[str],
                 filenames: List[str],
                 chunk_ids: List[int],
                 urls: List[str] = None,
                 titles: List[str] = None,
                 subtitles: List[str] = None,
                 chunk_text_only_list: List[str] = None,
                 file_types: List[str] = None,
                 namespace: str = None,
                 show_progress: bool = False):
    """
    Upsert vectors to Pinecone.
      - id = "{sanitized_doc_id}::chunk-{chunk_id}" (ASCII-safe for Pinecone requirements)
      - metadata = { "text": <full chunk with title/subtitle>, "doc_id": ..., "filename": ..., 
                     "chunk_id": ..., "url": ..., "title": ..., "subtitle": ..., "chunk_text_only": ... }
      - namespace: Optional namespace for dev/prod/language separation
    
    Note: "text" is used for embedding (includes title/subtitle for better retrieval),
          "chunk_text_only" is stored for display purposes (just the content).
          Vector IDs are sanitized (ASCII-only) but metadata preserves original doc_id.
    """
    vectors = []
    for local_idx, (emb, txt, doc_id, fn, cid) in enumerate(
        zip(embeddings, texts, doc_ids, filenames, chunk_ids)
    ):
        # Use document-based ID for easier debugging and reindexing
        # Pinecone requires vector IDs to be exactly 512 characters or less
        chunk_suffix = f"::chunk-{cid}"
        suffix_length = len(chunk_suffix)
        max_doc_part_length = 512 - suffix_length  # Reserve space for suffix
        
        # Sanitize doc_id (ASCII-safe)
        safe_doc_id = sanitize_doc_id_for_vector_id(doc_id)
        
        # If doc_id is too long, use hash instead (more reliable than truncation)
        if len(safe_doc_id) > max_doc_part_length:
            # Use hash of original doc_id (stable, deterministic)
            doc_hash = hashlib.md5(str(doc_id).encode('utf-8')).hexdigest()
            # Ensure hash fits within available space
            safe_doc_id = doc_hash[:max_doc_part_length]
        
        global_id = f"{safe_doc_id}{chunk_suffix}"
        
        # Absolute safety check: enforce 512 character limit
        # IMPORTANT: Preserve the chunk suffix integrity - only truncate the doc_id part
        if len(global_id) > 512:
            # Recalculate available space for doc_id part (suffix length might vary with large chunk IDs)
            actual_suffix_length = len(chunk_suffix)
            available_space = 512 - actual_suffix_length
            if available_space <= 0:
                # Edge case: suffix itself is too long (shouldn't happen with reasonable chunk IDs)
                print(f"[ERROR] Chunk suffix too long ({actual_suffix_length} chars): {chunk_suffix}")
                print(f"[ERROR] Skipping vector with doc_id: {doc_id[:100]}...")
                continue
            # Truncate only the doc_id part, preserving the full suffix
            safe_doc_id = safe_doc_id[:available_space]
            global_id = f"{safe_doc_id}{chunk_suffix}"
            # Final verification
            if len(global_id) > 512:
                print(f"[ERROR] Vector ID still too long after truncation: {len(global_id)} chars")
                print(f"[ERROR] Skipping vector with doc_id: {doc_id[:100]}...")
                continue
        
        # Ensure text is always a non-empty string (Pinecone requirement)
        text_value = str(txt) if txt else ""
        if not text_value:
            print(f"[WARN] Skipping vector with empty text: {global_id}")
            continue
        
        metadata = {
            "text": text_value,  # Full text with title/subtitle (used for embedding)
            "chunk_id": int(cid),
        }
        # Add doc_id if not None (Pinecone doesn't allow None values)
        if doc_id is not None:
            metadata["doc_id"] = str(doc_id)
        # Add filename if not None/empty
        if fn:
            metadata["filename"] = str(fn)
        # Add optional metadata (only if not None/empty)
        if urls and local_idx < len(urls) and urls[local_idx]:
            metadata["url"] = str(urls[local_idx])
        if titles and local_idx < len(titles) and titles[local_idx]:
            metadata["title"] = str(titles[local_idx])
        if subtitles and local_idx < len(subtitles) and subtitles[local_idx]:
            metadata["subtitle"] = str(subtitles[local_idx])
        if chunk_text_only_list and local_idx < len(chunk_text_only_list) and chunk_text_only_list[local_idx]:
            metadata["chunk_text_only"] = str(chunk_text_only_list[local_idx])
        if file_types and local_idx < len(file_types) and file_types[local_idx]:
            metadata["file_type"] = str(file_types[local_idx])
        
        # Validate metadata size (Pinecone has 40KB limit per metadata field)
        # We'll truncate text if needed, but keep it reasonably sized
        MAX_METADATA_VALUE_SIZE = 30000  # ~30KB to stay safely under 40KB limit
        text_bytes = text_value.encode('utf-8')
        if len(text_bytes) > MAX_METADATA_VALUE_SIZE:
            # Truncate by byte length, not character count (important for multibyte UTF-8)
            # Truncate to leave room for "..." suffix (3 bytes)
            truncated_bytes = text_bytes[:MAX_METADATA_VALUE_SIZE - 3]
            # Decode back to string, handling potential incomplete UTF-8 sequences at the end
            text_value = truncated_bytes.decode('utf-8', errors='ignore') + "..."
            metadata["text"] = text_value
        
        vector_data = {
            "id": global_id,
            "values": emb,
            "metadata": metadata,
        }
        vectors.append(vector_data)

    if vectors:
        if show_progress:
            namespace_str = f" (namespace: {namespace})" if namespace else ""
            print(f"[STEP] Upserting {len(vectors)} vectors to Pinecone{namespace_str}...")
        
        # Upsert with optional namespace
        try:
            if namespace:
                index.upsert(vectors=vectors, namespace=namespace)
            else:
                index.upsert(vectors=vectors)
            if show_progress:
                print(f"[OK] Successfully upserted {len(vectors)} vectors")
        except Exception as e:
            print(f"[ERROR] Failed to upsert vectors to Pinecone")
            print(f"[ERROR] Error type: {type(e).__name__}")
            print(f"[ERROR] Error message: {str(e)}")
            # Print details about first vector for debugging
            if vectors:
                first_vec = vectors[0]
                print(f"[DEBUG] First vector ID: {first_vec.get('id')}")
                print(f"[DEBUG] First vector metadata keys: {list(first_vec.get('metadata', {}).keys())}")
                print(f"[DEBUG] First vector metadata sample: {str(first_vec.get('metadata', {}))[:200]}")
            raise


# --- Main indexing routine ---

def index_haifa_data(prepared_dir: str,
                    paragraph_parquet: str,
                    paragraph_csv: str,
                    api_keys_path: str,
                    embedding_model_name: str,
                    index_name: str,
                    batch_size: int,
                    namespace: str = None):
    """
    Index Haifa municipality data into Pinecone.
    
    Args:
        namespace: Optional namespace for dev/prod/language separation (e.g., "dev", "prod", "hebrew", "arabic")
    """
    # 1) Pinecone init
    print("[STEP] Initializing Pinecone...")
    pinecone_api_key = load_pinecone_api_key(api_keys_path)
    pc = Pinecone(api_key=pinecone_api_key)
    print("[OK] Pinecone initialized")

    # 2) Embedding model
    embed_model = EmbeddingModel(embedding_model_name, verbose=True)

    # 3) Create / get index
    print(f"[STEP] Setting up index '{index_name}'...")
    create_index(pc, index_name, embed_model.dimension)
    index = pc.Index(index_name)
    print(f"[OK] Index '{index_name}' ready")

    # 4) Stream chunks and index in batches
    print(f"\n[STEP] Starting indexing process...")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Target index: '{index_name}'")
    if namespace:
        print(f"[INFO] Namespace: '{namespace}'")
    print()

    texts_batch: List[str] = []
    doc_ids_batch: List[str] = []
    filenames_batch: List[str] = []
    chunk_ids_batch: List[int] = []
    urls_batch: List[str] = []
    titles_batch: List[str] = []
    subtitles_batch: List[str] = []
    chunk_text_only_list_batch: List[str] = []
    file_types_batch: List[str] = []

    total = 0
    batch_num = 0

    chunk_iterator = iter_chunks(
        prepared_dir, paragraph_parquet, paragraph_csv,
        show_progress=True
    )
    
    for rec in chunk_iterator:
        texts_batch.append(rec["text"])  # Full text with title/subtitle (for embedding)
        doc_ids_batch.append(str(rec["doc_id"]) if rec["doc_id"] is not None else None)
        filenames_batch.append(rec["filename"])
        chunk_ids_batch.append(rec["chunk_id"])
        urls_batch.append(rec.get("url", ""))
        titles_batch.append(rec.get("title", ""))
        subtitles_batch.append(rec.get("subtitle", ""))
        chunk_text_only_list_batch.append(rec.get("chunk_text_only", ""))
        file_types_batch.append(rec.get("file_type", "html"))

        if len(texts_batch) >= batch_size:
            batch_num += 1
            print(f"\n[STEP] Processing batch {batch_num} ({len(texts_batch)} chunks)...")
            
            embs = embed_model.embed(texts_batch, show_progress=True)
            upsert_index(
                index=index,
                embeddings=embs,
                texts=texts_batch,
                doc_ids=doc_ids_batch,
                filenames=filenames_batch,
                chunk_ids=chunk_ids_batch,
                urls=urls_batch,
                titles=titles_batch,
                subtitles=subtitles_batch,
                chunk_text_only_list=chunk_text_only_list_batch,
                file_types=file_types_batch,
                namespace=namespace,
                show_progress=True,
            )
            total += len(texts_batch)
            print(f"[INFO] Batch {batch_num} complete. Total vectors indexed: {total:,}")

            # reset batch
            texts_batch, doc_ids_batch, filenames_batch, chunk_ids_batch = [], [], [], []
            urls_batch, titles_batch, subtitles_batch, chunk_text_only_list_batch, file_types_batch = [], [], [], [], []

    # Final remainder
    if texts_batch:
        batch_num += 1
        print(f"\n[STEP] Processing final batch {batch_num} ({len(texts_batch)} chunks)...")
        
        embs = embed_model.embed(texts_batch, show_progress=True)
        upsert_index(
            index=index,
            embeddings=embs,
            texts=texts_batch,
            doc_ids=doc_ids_batch,
            filenames=filenames_batch,
            chunk_ids=chunk_ids_batch,
            urls=urls_batch,
            titles=titles_batch,
            subtitles=subtitles_batch,
            chunk_text_only_list=chunk_text_only_list_batch,
            file_types=file_types_batch,
            namespace=namespace,
            show_progress=True,
        )
        total += len(texts_batch)
        print(f"[INFO] Final batch complete. Total vectors indexed: {total:,}")

    print(f"\n{'='*60}")
    print(f"[OK] Finished indexing Haifa municipality data!")
    print(f"[INFO] Total vectors indexed: {total:,}")
    print(f"[INFO] Total batches processed: {batch_num}")
    print(f"{'='*60}")


# --- CLI ---

def parse_args():
    parser = argparse.ArgumentParser(
        description="Haifa Municipality Data Indexing"
    )
    parser.add_argument("--prepared_dir", type=str, default=DEFAULT_PREPARED_DIR,
                        help="Directory where data_preparation.py wrote its outputs.")
    parser.add_argument("--paragraph_parquet", type=str, default=DEFAULT_PARAGRAPH_PARQUET,
                        help="Name of the paragraph Parquet file inside prepared_dir (supports config-based names).")
    parser.add_argument("--paragraph_csv", type=str, default=DEFAULT_PARAGRAPH_CSV,
                        help="Fallback CSV file name inside prepared_dir (supports config-based names).")
    parser.add_argument("--config", type=str, default=None,
                        help="Config suffix to use (e.g., 'chunk1000_overlap200'). If provided, overrides paragraph_parquet/csv.")
    parser.add_argument("--api_keys_path", type=str, default=DEFAULT_API_KEYS_PATH,
                        help="Path to api_keys.json with 'PINECONE_API_KEY' key.")
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                        help="SentenceTransformer model name.")
    parser.add_argument("--index_name", type=str, default=DEFAULT_INDEX_NAME,
                        help="Pinecone index name.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for embedding & upsert.")
    parser.add_argument("--namespace", type=str, default=None,
                        help="Optional namespace for dev/prod/language separation (e.g., 'dev', 'prod', 'hebrew', 'arabic')")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # If config is provided, use it to construct filenames
    if args.config:
        paragraph_parquet = f"haifa_paragraph_index_config_{args.config}.parquet"
        paragraph_csv = f"haifa_paragraph_index_config_{args.config}.csv"
    else:
        paragraph_parquet = args.paragraph_parquet
        paragraph_csv = args.paragraph_csv
    
    index_haifa_data(
        prepared_dir=args.prepared_dir,
        paragraph_parquet=paragraph_parquet,
        paragraph_csv=paragraph_csv,
        api_keys_path=args.api_keys_path,
        embedding_model_name=args.embedding_model,
        index_name=args.index_name,
        batch_size=args.batch_size,
        namespace=args.namespace,
    )


if __name__ == "__main__":
    main()

