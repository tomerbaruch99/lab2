import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from reranker import rerank
from bm25_index import bm25_search

LEGALBERT = SentenceTransformer("nlpaueb/legal-bert-base-uncased")
CLAUSES = pd.read_parquet("cuad_prepared_data/cuad_long_clauses.parquet")
EMB = np.load("data/CUAD_v1/embeddings.npy").astype("float32")

# Shared FAISS index
import faiss
index = faiss.IndexFlatL2(EMB.shape[1])
index.add(EMB)

def retrieve(query, k_dense=8, k_bm25=8, final_k=5):
    # Dense retrieval
    qvec = LEGALBERT.encode([query])
    dist, idx = index.search(qvec, k_dense)
    dense_results = CLAUSES.iloc[idx[0]]["context"].tolist()

    # BM25 retrieval
    sparse_results = bm25_search(query, k=k_bm25)["context"].tolist()

    # Merge candidates
    candidates = list(set(dense_results + sparse_results))

    # Rerank final list
    ranked = rerank(query, candidates)
    return ranked[:final_k]
