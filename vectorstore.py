import numpy as np
import faiss
import pandas as pd

class ClauseVectorStore:
    def __init__(self, df, embeddings):
        self.df = df
        self.embeddings = embeddings.astype("float32")
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(query_embedding, k)
        results = self.df.iloc[indices[0]]
        return results, distances[0]

def load_vectorstore():
    df = pd.read_csv("data/cuad_clauses_extracted.csv")
    embeddings = np.load("data/cuad_embeddings.npy")
    return ClauseVectorStore(df, embeddings)
