"""
Haifa Municipality – Vector Indexing for RAG
============================================

This script:
1. Loads processed chunks from data_preparation.py (haifa_rag_chunks.parquet)
2. Generates embeddings using SentenceTransformer
3. Indexes vectors into Pinecone with full metadata:
       - namespace (per-chunk)
       - doc_type
       - chunking_strategy
       - title/subtitle
       - url
       - chunk_text_only
4. Ensures Pinecone-safe vector IDs with hashing when necessary

Usage:
    python indexing.py \
        --prepared_file ./scrape_and_prepare_data/haifa_prepared_data/haifa_rag_chunks.parquet \
        --api_keys_path api_keys.json \
        --embedding_model paraphrase-multilingual-MiniLM-L12-v2 \
        --index_name haifa-municipality-rag \
        --batch_size 128
"""

import os
import argparse
import hashlib
import json
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


# ============================================================
# Load chunks from parquet
# ============================================================

def iter_chunks(prepared_file: str, show_progress: bool = True) -> Iterable[Dict]:
    """
    Load chunks from the unified haifa_rag_chunks.parquet file.

    Required columns in parquet:
        - text
        - chunk_text_only
        - doc_id
        - chunk_id
        - url
        - title
        - subtitle
        - doc_type
        - namespace
        - chunking_strategy
        - links (JSON string)
    """

    if not os.path.exists(prepared_file):
        raise FileNotFoundError(f"Prepared file not found: {prepared_file}")

    print(f"[STEP] Loading prepared chunks from {prepared_file} ...")
    df = pd.read_parquet(prepared_file)
    total = len(df)
    print(f"[OK] Loaded {total:,} rows")

    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=total, desc="Reading chunks", unit="chunk")

    for _, row in iterator:
        text = str(row.get("text") or "").strip()
        if not text:
            continue

        yield {
            "text": text,
            "chunk_text_only": row.get("chunk_text_only", ""),
            "doc_id": row.get("doc_id", ""),
            "chunk_id": int(row.get("chunk_id", 0)),
            "url": row.get("url", ""),
            "title": row.get("title", ""),
            "subtitle": row.get("subtitle", ""),
            "doc_type": row.get("doc_type", "unknown"),
            "namespace": row.get("namespace", "general"),
            "chunking_strategy": row.get("chunking_strategy", "unknown"),
            "links": row.get("links", "[]"),
        }


# ============================================================
# Pinecone vector ID helper
# ============================================================

def sanitize_vector_id(doc_id: str, chunk_id: int) -> str:
    """
    Produce Pinecone-safe ASCII-only vector IDs.
    Format:
        <ascii-doc-id>::chunk-<id>

    Falls back to MD5 hash if needed.
    """

    suffix = f"::chunk-{chunk_id}"
    max_total_length = 512

    # URL-encode doc_id to ensure ASCII
    safe = quote(str(doc_id), safe='')

    # If too long, hash it
    if len(safe) + len(suffix) > max_total_length:
        hashed = hashlib.md5(str(doc_id).encode()).hexdigest()
        return f"{hashed}{suffix}"

    return f"{safe}{suffix}"


# ============================================================
# Metadata size management
# ============================================================

def truncate_metadata(metadata: Dict, max_bytes: int = 40000) -> Dict:
    """
    Truncate metadata fields to ensure total size is under max_bytes.
    Pinecone limit is 40960 bytes, we use 40000 for safety margin.
    
    Prioritizes keeping: doc_id, chunk_id, namespace, doc_type, url
    Truncates: text, chunk_text_only, title, subtitle, links
    """
    
    # Calculate current size
    current_size = len(json.dumps(metadata, ensure_ascii=False).encode('utf-8'))
    
    if current_size <= max_bytes:
        return metadata
    
    # Fields to truncate (in order of priority - truncate less important ones first)
    truncate_fields = ['links', 'subtitle', 'title', 'chunk_text_only', 'text']
    
    # Create a copy to modify
    truncated = metadata.copy()
    
    # Truncate fields until we're under the limit
    for field in truncate_fields:
        if field not in truncated:
            continue
            
        # Calculate size without this field
        test_meta = truncated.copy()
        test_meta[field] = ""
        base_size = len(json.dumps(test_meta, ensure_ascii=False).encode('utf-8'))
        
        # Calculate how much space we have for this field
        available_bytes = max_bytes - base_size - 100  # 100 byte safety margin
        
        if available_bytes <= 0:
            # No space, remove field entirely
            truncated[field] = ""
            continue
        
        # Truncate field to fit
        field_value = str(truncated[field])
        if len(field_value.encode('utf-8')) > available_bytes:
            # Truncate by bytes, not characters (to handle multi-byte UTF-8)
            truncated[field] = field_value.encode('utf-8')[:available_bytes].decode('utf-8', errors='ignore')
            # Remove incomplete characters at the end
            while truncated[field] and len(truncated[field].encode('utf-8')) > available_bytes:
                truncated[field] = truncated[field][:-1]
        
        # Check if we're under limit now
        current_size = len(json.dumps(truncated, ensure_ascii=False).encode('utf-8'))
        if current_size <= max_bytes:
            break
    
    return truncated


# ============================================================
# Upsert to Pinecone
# ============================================================

def upsert_batch(index, embeddings, batch_records, show_progress=False):
    """
    Upserts a batch of vectors into Pinecone.
    Each entry in batch_records contains:
        - text
        - chunk_text_only
        - doc_id
        - chunk_id
        - namespace
        - doc_type
        - chunking_strategy
        - title
        - subtitle
        - url
        - links
    """

    vectors = []

    for emb, rec in zip(embeddings, batch_records):

        vector_id = sanitize_vector_id(rec["doc_id"], rec["chunk_id"])
        text_value = rec["text"]

        metadata = {
            "text": text_value,
            "chunk_text_only": rec["chunk_text_only"],
            "doc_id": rec["doc_id"],
            "chunk_id": rec["chunk_id"],
            "doc_type": rec["doc_type"],
            "namespace": rec["namespace"],
            "chunking_strategy": rec["chunking_strategy"],
            "title": rec["title"],
            "subtitle": rec["subtitle"],
            "url": rec["url"],
            "links": rec["links"],
        }
        
        # Truncate metadata to ensure it's under Pinecone's 40KB limit
        metadata = truncate_metadata(metadata)

        vectors.append({
            "id": vector_id,
            "values": emb,
            "metadata": metadata
        })

    # Upsert by namespace (per chunk!)
    for ns in set(rec["namespace"] for rec in batch_records):
        sub_vectors = [v for v in vectors if v["metadata"]["namespace"] == ns]

        if show_progress:
            print(f"[PINECONE] Upserting {len(sub_vectors)} vectors into namespace '{ns}'")

        index.upsert(vectors=sub_vectors, namespace=ns)


# ============================================================
# Main indexing routine
# ============================================================

def index_haifa(
    prepared_file: str,
    api_keys_path: str,
    embedding_model_name: str,
    index_name: str,
    batch_size: int
):
    # 1. Init Pinecone
    key = load_pinecone_api_key(api_keys_path)
    pc = Pinecone(api_key=key)

    # 2. Load embedding model (force CPU to avoid CUDA compatibility issues)
    embed_model = EmbeddingModel(embedding_model_name, device="cpu", verbose=True)

    # 3. Create/get index
    create_index(pc, index_name, embed_model.dimension)
    index = pc.Index(index_name)

    # 4. Iterate over chunks
    batch_records = []
    texts_batch = []

    print("\n[START] Indexing...")
    for rec in iter_chunks(prepared_file, show_progress=True):
        batch_records.append(rec)
        texts_batch.append(rec["text"])

        if len(batch_records) >= batch_size:
            embeddings = embed_model.embed(texts_batch, show_progress=False)
            upsert_batch(index, embeddings, batch_records, show_progress=True)

            batch_records = []
            texts_batch = []

    # Final remainder
    if batch_records:
        embeddings = embed_model.embed(texts_batch, show_progress=False)
        upsert_batch(index, embeddings, batch_records, show_progress=True)

    print("[DONE] Indexing complete!")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prepared_file",
        type=str,
        required=True,
        help="Path to haifa_rag_chunks.parquet"
    )

    parser.add_argument(
        "--api_keys_path",
        type=str,
        default=DEFAULT_API_KEYS_PATH
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL
    )

    parser.add_argument(
        "--index_name",
        type=str,
        default=DEFAULT_INDEX_NAME
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )

    return parser.parse_args()


def main():
    args = parse_args()

    index_haifa(
        prepared_file=args.prepared_file,
        api_keys_path=args.api_keys_path,
        embedding_model_name=args.embedding_model,
        index_name=args.index_name,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
