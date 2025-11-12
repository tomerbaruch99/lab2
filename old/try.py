import numpy as np, pandas as pd
E = np.load("processed_data/dense_embeddings.npy", mmap_mode="r")
df = pd.read_parquet("cuad_prepared_data/cuad_long_clauses.parquet", columns=["context"])
print("Emb shape:", E.shape)                 # (N, 384) for MiniLM
print("Rows in parquet:", len(df))           # N must match
print("First row L2 norm:", float(np.linalg.norm(E[0])))
