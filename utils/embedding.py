"""
Embedding model wrapper shared across the RAG system.
"""

from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper for SentenceTransformer embedding model.
    
    Used for both indexing and retrieval to ensure consistent embeddings.
    """
    
    def __init__(self, model_name: str, device: str = "cpu", verbose: bool = False):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model
            device: Device to use ("cpu" or "cuda")
            verbose: Whether to print loading progress
        """
        if verbose:
            print(f"[STEP] Loading embedding model '{model_name}'...")
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        if verbose:
            print(f"[INFO] Embedding dimension: {self.dimension}")
            print(f"[INFO] Using device: {device}")
    
    def embed(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """
        Embed a list of texts.
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar
        
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []
        if show_progress and len(texts) > 1:
            print(f"[STEP] Embedding {len(texts)} texts...")
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress)
        return embs.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.
        
        Args:
            query: Query text to embed
        
        Returns:
            Embedding vector as a list of floats
        """
        return self.embed([query])[0]

