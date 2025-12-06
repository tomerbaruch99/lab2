"""
Embedding model wrapper shared across the RAG system.
"""

from typing import List

import torch
from sentence_transformers import SentenceTransformer


def is_cuda_compatible() -> bool:
    """
    Check if CUDA is available and compatible with the current PyTorch installation.
    
    Returns:
        True if CUDA is available and compatible, False otherwise.
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        # Try to get CUDA capability
        if torch.cuda.device_count() > 0:
            capability = torch.cuda.get_device_capability(0)
            # PyTorch typically supports compute capability 7.0+
            # Older GPUs (like Tesla M60 with 5.2) are not supported
            if capability[0] < 7:
                return False
            # Try a simple CUDA operation to verify it works
            try:
                test_tensor = torch.tensor([1.0], device="cuda")
                _ = test_tensor * 2
                del test_tensor
                torch.cuda.empty_cache()
                return True
            except (RuntimeError, Exception) as e:
                # CUDA operation failed - GPU is not compatible
                return False
    except Exception:
        # If any error occurs, CUDA is not usable
        return False
    
    return False


class EmbeddingModel:
    """
    Wrapper for SentenceTransformer embedding model.
    
    Used for both indexing and retrieval to ensure consistent embeddings.
    """
    
    def __init__(self, model_name: str, device: str = None, verbose: bool = False):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model
            device: Device to use ("cpu" or "cuda"). If None, automatically uses CUDA if compatible.
            verbose: Whether to print loading progress
        """
        if device is None:
            # Only use CUDA if it's actually compatible
            device = "cuda" if is_cuda_compatible() else "cpu"
        
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

