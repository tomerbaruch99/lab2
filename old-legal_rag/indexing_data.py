import os
import json
import argparse
from typing import List, Dict, Iterable

import pandas as pd
import torch
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


# ===============================
# Defaults (tweak as you like)
# ===============================

DEFAULT_PREPARED_DIR = "./cuad_prepared_data"
DEFAULT_PARAGRAPH_PARQUET = "cuad_paragraph_index.parquet"
DEFAULT_PARAGRAPH_CSV = "cuad_paragraph_index.csv"

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_INDEX_NAME = "contracts-recursive-index"
DEFAULT_API_KEYS_PATH = "api_keys.json"


# ===============================
# Pinecone helpers
# ===============================

def load_pinecone_api_key(api_keys_path: str) -> str:
    """Load Pinecone API key from env or from api_keys.json (same pattern as ToS project)."""
    env_key = os.getenv("PINECONE_API_KEY")
    if env_key:
        return env_key

    if not os.path.exists(api_keys_path):
        raise FileNotFoundError(
            f"api_keys.json not found at {api_keys_path} and PINECONE_API_KEY env var not set."
        )

    with open(api_keys_path, "r", encoding="utf-8") as f:
        api_keys = json.load(f)

    if "PINECONE_API_KEY" not in api_keys:
        raise KeyError("Key 'PINECONE_API_KEY' missing from api_keys.json")

    return api_keys["PINECONE_API_KEY"]


def create_index(pc: Pinecone, index_name: str, dimension: int):
    """Identical spirit to your ToS create_index()."""
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"[INFO] Creating index '{index_name}' (dim={dimension}, metric='cosine')...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"[INFO] Index '{index_name}' already exists.")


# ===============================
# Embedding model wrapper
# ===============================

class EmbeddingModel:
    def __init__(self, model_name: str):
        print(f"[STEP] Loading embedding model '{model_name}'...")
        # Force CPU usage to avoid CUDA compatibility issues
        device = "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[INFO] Embedding dimension: {self.dimension}")
        print(f"[INFO] Using device: {device}")

    def embed(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        if not texts:
            return []
        if show_progress:
            print(f"[STEP] Embedding {len(texts)} texts...")
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress)
        return embs.tolist()


# ===============================
# Data iterator – paragraphs only
# ===============================

def iter_paragraph_chunks(prepared_dir: str,
                          paragraph_parquet: str,
                          paragraph_csv: str,
                          split_filter: str = None,
                          long_clauses_path: str = None,
                          show_progress: bool = True) -> Iterable[Dict]:
    """
    Yield chunks from CUAD paragraph index, optionally filtered by split.
    Each yielded dict has:
      - text
      - doc_id (contract ID)
      - filename
      - chunk_id
      - start_char
    
    Args:
        split_filter: If "train", "val", or "test", only yield chunks from that split.
                     Requires long_clauses_path to get split information.
        long_clauses_path: Path to cuad_long_clauses.parquet to get split info.
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

    # Load split information if filtering by split
    train_filenames = set()
    if split_filter and long_clauses_path:
        if os.path.exists(long_clauses_path):
            print(f"[STEP] Loading split information from {long_clauses_path}...")
            long_df = pd.read_parquet(long_clauses_path)
            if "split" in long_df.columns:
                train_df = long_df[long_df["split"] == split_filter]
                train_filenames = set(train_df["filename"].unique())
                print(f"[INFO] Found {len(train_filenames)} {split_filter} documents")
            else:
                print(f"[WARN] No 'split' column in long_clauses, cannot filter by split")
        else:
            print(f"[WARN] long_clauses file not found at {long_clauses_path}, cannot filter by split")

    print(f"[STEP] Loading chunks from {path} ...")

    if ext == ".parquet":
        df = pd.read_parquet(path)
        total_rows = len(df)
        print(f"[INFO] Found {total_rows:,} rows in parquet file")
        
        iterator = df.iterrows()
        if show_progress:
            iterator = tqdm(iterator, total=total_rows, desc="Reading chunks", unit="chunk")
        
        for _, row in iterator:
            # Filter by split if requested
            if split_filter and train_filenames:
                filename = row.get("filename", "")
                if filename not in train_filenames:
                    continue
            
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            yield {
                "text": text,
                "doc_id": row.get("doc_id"),
                "filename": row.get("filename"),
                "chunk_id": int(row.get("chunk_id")),
                "start_char": int(row.get("start_char")),
            }

    else:  # CSV
        # For CSV, we need to count rows first for progress bar
        if show_progress:
            print(f"[INFO] Counting rows in CSV file...")
            total_rows = sum(1 for _ in pd.read_csv(path, chunksize=50_000, usecols=[0]))
            print(f"[INFO] Found {total_rows:,} rows in CSV file")
            pbar = tqdm(total=total_rows, desc="Reading chunks", unit="chunk")
        else:
            pbar = None
        
        processed = 0
        for chunk in pd.read_csv(path, chunksize=50_000):
            for _, row in chunk.iterrows():
                text = str(row.get("text") or "").strip()
                if not text:
                    if pbar:
                        pbar.update(1)
                    continue
                processed += 1
                if pbar:
                    pbar.update(1)
                yield {
                    "text": text,
                    "doc_id": row.get("doc_id"),
                    "filename": row.get("filename"),
                    "chunk_id": int(row.get("chunk_id")),
                    "start_char": int(row.get("start_char")),
                }
        
        if pbar:
            pbar.close()


# ===============================
# Upsert helper
# ===============================

def upsert_index(index,
                 embeddings: List[List[float]],
                 texts: List[str],
                 doc_ids: List[str],
                 filenames: List[str],
                 chunk_ids: List[int],
                 batch_base_id: int,
                 show_progress: bool = False):
    """
    Mirrors your original upsert_index:
      - id = global running index (string)
      - metadata = { "text": <chunk>, "doc_id": ..., "filename": ..., "chunk_id": ... }
    """
    vectors = []
    for local_idx, (emb, txt, doc_id, fn, cid) in enumerate(
        zip(embeddings, texts, doc_ids, filenames, chunk_ids)
    ):
        global_id = batch_base_id + local_idx
        metadata = {
            "text": txt,
            "doc_id": str(doc_id) if doc_id is not None else None,
            "filename": fn,
            "chunk_id": int(cid),
        }
        vectors.append(
            {
                "id": str(global_id),
                "values": emb,
                "metadata": metadata,
            }
        )

    if vectors:
        if show_progress:
            print(f"[STEP] Upserting {len(vectors)} vectors to Pinecone...")
        index.upsert(vectors=vectors)
        if show_progress:
            print(f"[OK] Successfully upserted {len(vectors)} vectors")


# ===============================
# Main indexing routine
# ===============================

def index_contracts(prepared_dir: str,
                    paragraph_parquet: str,
                    paragraph_csv: str,
                    api_keys_path: str,
                    embedding_model_name: str,
                    index_name: str,
                    batch_size: int,
                    split_filter: str = None,
                    long_clauses_path: str = None):
    """
    Index contracts into Pinecone.
    
    Args:
        split_filter: If "train", "val", or "test", only index chunks from that split.
                     Requires long_clauses_path to get split information.
        long_clauses_path: Path to cuad_long_clauses.parquet to get split info.
    """
    # 1) Pinecone init
    print("[STEP] Initializing Pinecone...")
    pinecone_api_key = load_pinecone_api_key(api_keys_path)
    pc = Pinecone(api_key=pinecone_api_key)
    print("[OK] Pinecone initialized")

    # 2) Embedding model
    embed_model = EmbeddingModel(embedding_model_name)

    # 3) Create / get index
    print(f"[STEP] Setting up index '{index_name}'...")
    create_index(pc, index_name, embed_model.dimension)
    index = pc.Index(index_name)
    print(f"[OK] Index '{index_name}' ready")

    # 4) Stream chunks and index in batches
    print(f"\n[STEP] Starting indexing process...")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Target index: '{index_name}'")
    if split_filter:
        print(f"[INFO] Filtering by split: '{split_filter}' (only indexing {split_filter} documents)")
    print()

    texts_batch: List[str] = []
    doc_ids_batch: List[str] = []
    filenames_batch: List[str] = []
    chunk_ids_batch: List[int] = []

    total = 0
    batch_num = 0

    # Use tqdm for main progress tracking
    chunk_iterator = iter_paragraph_chunks(
        prepared_dir, paragraph_parquet, paragraph_csv,
        split_filter=split_filter,
        long_clauses_path=long_clauses_path,
        show_progress=True
    )
    
    for rec in chunk_iterator:
        texts_batch.append(rec["text"])
        doc_ids_batch.append(str(rec["doc_id"]) if rec["doc_id"] is not None else None)
        filenames_batch.append(rec["filename"])
        chunk_ids_batch.append(rec["chunk_id"])

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
                batch_base_id=total,
                show_progress=True,
            )
            total += len(texts_batch)
            print(f"[INFO] Batch {batch_num} complete. Total vectors indexed: {total:,}")

            # reset batch
            texts_batch, doc_ids_batch, filenames_batch, chunk_ids_batch = [], [], [], []

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
            batch_base_id=total,
            show_progress=True,
        )
        total += len(texts_batch)
        print(f"[INFO] Final batch complete. Total vectors indexed: {total:,}")

    print(f"\n{'='*60}")
    print(f"[OK] Finished indexing contracts!")
    print(f"[INFO] Total vectors indexed: {total:,}")
    print(f"[INFO] Total batches processed: {batch_num}")
    print(f"{'='*60}")


# ===============================
# CLI
# ===============================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CUAD contracts indexing"
    )
    parser.add_argument("--prepared_dir", type=str, default=DEFAULT_PREPARED_DIR,
                        help="Directory where data_preparation.py wrote its outputs.")
    parser.add_argument("--paragraph_parquet", type=str, default=DEFAULT_PARAGRAPH_PARQUET,
                        help="Name of the paragraph Parquet file inside prepared_dir.")
    parser.add_argument("--paragraph_csv", type=str, default=DEFAULT_PARAGRAPH_CSV,
                        help="Fallback CSV file name inside prepared_dir.")
    parser.add_argument("--api_keys_path", type=str, default=DEFAULT_API_KEYS_PATH,
                        help="Path to api_keys.json with 'PINECONE_API_KEY' key.")
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                        help="SentenceTransformer model name.")
    parser.add_argument("--index_name", type=str, default=DEFAULT_INDEX_NAME,
                        help="Pinecone index name.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for embedding & upsert.")
    parser.add_argument("--split_filter", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Only index documents from this split (requires --long_clauses_path).")
    parser.add_argument("--long_clauses_path", type=str, default=None,
                        help="Path to cuad_long_clauses.parquet for split filtering.")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Auto-detect long_clauses path if split_filter is set but path not provided
    long_clauses_path = args.long_clauses_path
    if args.split_filter and not long_clauses_path:
        long_clauses_path = os.path.join(args.prepared_dir, "cuad_long_clauses.parquet")
        if not os.path.exists(long_clauses_path):
            print(f"[WARN] Split filter '{args.split_filter}' requested but long_clauses not found at {long_clauses_path}")
            print(f"[WARN] Proceeding without split filtering...")
            args.split_filter = None

    index_contracts(
        prepared_dir=args.prepared_dir,
        paragraph_parquet=args.paragraph_parquet,
        paragraph_csv=args.paragraph_csv,
        api_keys_path=args.api_keys_path,
        embedding_model_name=args.embedding_model,
        index_name=args.index_name,
        batch_size=args.batch_size,
        split_filter=args.split_filter,
        long_clauses_path=long_clauses_path,
    )


if __name__ == "__main__":
    main()
