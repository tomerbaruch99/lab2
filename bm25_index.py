from rank_bm25 import BM25Okapi
import pandas as pd

def build_bm25(input_csv="data/CUAD_v1/clauses.csv"):
    df = pd.read_csv(input_csv)
    tokenized = [t.lower().split() for t in df["clause_text"]]
    bm25 = BM25Okapi(tokenized)
    return df, bm25

def bm25_search(query, k=10, input_csv="data/CUAD_v1/clauses.csv"):
    df, bm25 = build_bm25(input_csv)
    scores = bm25.get_scores(query.lower().split())
    top_indices = scores.argsort()[-k:][::-1]
    return df.iloc[top_indices]

if __name__ == "__main__":
    print(bm25_search("non compete"))