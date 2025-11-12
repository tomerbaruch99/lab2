import os, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

PARQUET = "cuad_prepared_data/cuad_long_clauses.parquet"
OUT = "processed_data/dense_embeddings.npy"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim

def main(batch_size=64, shard_size=20000, text_col="context"):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_parquet(PARQUET, columns=[text_col])
    texts = df[text_col].fillna("").astype(str).tolist()
    n = len(texts)

    model = SentenceTransformer(MODEL, device="cpu")
    dim = model.get_sentence_embedding_dimension()

    # Preallocate memmap for low RAM usage
    embs = np.memmap(OUT, dtype="float32", mode="w+", shape=(n, dim))

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
    print(f"[OK] Saved embeddings to {OUT}")

if __name__ == "__main__":
    main()
