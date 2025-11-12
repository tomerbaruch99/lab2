# embeddings_jina.py
from pathlib import Path
import os, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PARQUET = "cuad_prepared_data/cuad_long_clauses.parquet"
OUTPUT_NPY    = "processed_data/jina_embeddings.npy"

def create_embeddings(batch_size=32, shard_size=15000):
    df = pd.read_parquet(INPUT_PARQUET, columns=["context"])
    texts = df["context"].fillna("").astype(str).tolist()
    n = len(texts)

    model = SentenceTransformer(
        "jinaai/jina-embeddings-v2-base-en",
        device="cpu",
        trust_remote_code=True
    )

    dim = model.get_sentence_embedding_dimension()
    os.makedirs(os.path.dirname(OUTPUT_NPY), exist_ok=True)
    embs = np.memmap(OUTPUT_NPY, dtype="float32", mode="w+", shape=(n, dim))

    start = 0
    while start < n:
        end = min(start + shard_size, n)
        shard = texts[start:end]
        vecs = model.encode(
            shard, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True, device="cpu"
        ).astype("float32")
        embs[start:end] = vecs
        embs.flush()
        print(f"[✓] Encoded {end}/{n}")
        start = end

    del embs
    print(f"[OK] Saved {OUTPUT_NPY}")

if __name__ == "__main__":
    create_embeddings()
