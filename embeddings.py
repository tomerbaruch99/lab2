import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "jinaai/jina-embeddings-v2-base-en"
output_jina_npy="processed_data/jina_embeddings.npy"
output_legalbert_npy="processed_data/legalbert_embeddings.npy"
# MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
# MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def create_embeddings(input_parquet="cuad_prepared_data/cuad_long_clauses.parquet", output_npy=output_jina_npy):
    df = pd.read_parquet(input_parquet)
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(df["context"].tolist(), show_progress_bar=True)
    np.save(output_npy, embeddings)
    print(f"[✓] Saved embeddings to {output_npy}")

if __name__ == "__main__":
    create_embeddings()
