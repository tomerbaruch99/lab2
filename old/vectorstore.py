import numpy as np
import faiss
import pandas as pd

class ClauseVectorStore:
    def __init__(self, df: pd.DataFrame, embeddings: np.ndarray):
        self.df = df
        self.embeddings = embeddings.astype("float32")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])  # cosine via IP (assumes L2-normalized)
        self.index.add(self.embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 5):
        distances, indices = self.index.search(query_embedding, k)
        results = self.df.iloc[indices[0]]
        return results, distances[0]

def load_vectorstore(df_path="cuad_prepared_data/cuad_long_clauses.parquet",
                     emb_path="processed_data/jina_embeddings.npy"):
        df = pd.read_parquet(df_path)
        embeddings = np.load(emb_path)
        return ClauseVectorStore(df, embeddings)
