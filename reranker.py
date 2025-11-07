from FlagEmbedding import FlagReranker

# best legal-aware reranker model
RERANK_MODEL = "BAAI/bge-reranker-base"
reranker = FlagReranker(RERANK_MODEL)

def rerank(query, texts):
    pairs = [(query, t) for t in texts]
    scores = reranker.compute_score(pairs)
    sorted_pairs = sorted(zip(scores, texts), reverse=True)
    return [t for _, t in sorted_pairs]
