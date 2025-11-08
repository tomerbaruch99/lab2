from rank_bm25 import BM25Okapi
import pandas as pd

def build_bm25(input_parquet="cuad_prepared_data/cuad_long_clauses.parquet"):
    df = pd.read_parquet(input_parquet)
    tokenized = [t.lower().split() for t in df["context"]]
    bm25 = BM25Okapi(tokenized)
    return df, bm25

def bm25_search(query, k=10, input_parquet="cuad_prepared_data/cuad_long_clauses.parquet"):
    df, bm25 = build_bm25(input_parquet)
    scores = bm25.get_scores(query.lower().split())
    top_indices = scores.argsort()[-k:][::-1]
    return df.iloc[top_indices]

if __name__ == "__main__":
    print(bm25_search("non compete"))