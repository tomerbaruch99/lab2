import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "jinaai/jina-embeddings-v2-base-en"
output_jina_npy="data/CUAD_v1/jina_embeddings.npy"
output_legalbert_npy="data/CUAD_v1/legalbert_embeddings.npy"
# MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
# MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def create_embeddings(input_csv="data/CUAD_v1/clauses.csv", output_npy=output_jina_npy"):
    df = pd.read_csv(input_csv)
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(df["clause_text"].tolist(), show_progress_bar=True)
    np.save(output_npy, embeddings)
    print(f"[✓] Saved embeddings to {output_npy}")

if __name__ == "__main__":
    create_embeddings()
