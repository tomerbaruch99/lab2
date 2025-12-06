"""
Smart Page Finder - A Lightweight URL Recommender Based on User Query

When the user asks a question, this tool suggests the most relevant pages
on the official Haifa website based on semantic similarity.
"""

import csv
import os
import sys
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Import directly from modules to avoid circular dependencies
project_root = Path(__file__).parent.parent
embedding_path = project_root / "utils" / "embedding.py"
spec = importlib.util.spec_from_file_location("embedding", embedding_path)
embedding_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embedding_module)
EmbeddingModel = embedding_module.EmbeddingModel

config_path = project_root / "utils" / "config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
DEFAULT_EMBEDDING_MODEL = config_module.DEFAULT_EMBEDDING_MODEL
DEFAULT_TOP_K = config_module.DEFAULT_TOP_K


class SmartPageFinder:
    """
    Finds relevant pages from Haifa municipality website based on user query.
    """
    
    def __init__(
        self,
        page_index_path: str = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = DEFAULT_TOP_K
    ):
        """
        Initialize Smart Page Finder.
        
        Args:
            page_index_path: Path to page_index.csv. If None, looks in scrape_and_prepare_data/
            embedding_model_name: Name of the embedding model (must match the one used to build index)
            top_k: Number of top pages to return
        """
        # Set default path if not provided
        if page_index_path is None:
            project_root = Path(__file__).parent.parent
            page_index_path = project_root / "scrape_and_prepare_data" / "page_index.csv"
        
        self.page_index_path = Path(page_index_path)
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k
        
        # Load embedding model (force CPU to avoid CUDA compatibility issues)
        self.embed_model = EmbeddingModel(embedding_model_name, device="cpu", verbose=False)
        
        # Load page index
        self.pages = []
        self.embeddings = []
        self._load_index()
    
    def _load_index(self):
        """Load page index from CSV file."""
        if not self.page_index_path.exists():
            raise FileNotFoundError(
                f"Page index not found at {self.page_index_path}. "
                f"Please run build_page_index.py first."
            )
        
        print(f"[INFO] Loading page index from {self.page_index_path}...")
        
        with open(self.page_index_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.pages.append({
                    'title': row['title'],
                    'subtitle': row['subtitle'],
                    'url': row['url']
                })
                # Parse embedding from CSV string
                embedding = [float(x) for x in row['embedding'].split(',')]
                self.embeddings.append(embedding)
        
        self.embeddings = np.array(self.embeddings)
        print(f"[INFO] Loaded {len(self.pages)} pages")
    
    def find_relevant_pages(self, query: str, top_k: int = None) -> List[Dict[str, str]]:
        """
        Find most relevant pages for a user query.
        
        Args:
            query: User query text
            top_k: Number of top pages to return (defaults to self.top_k)
        
        Returns:
            List of dictionaries with 'title', 'subtitle', 'url', and 'score' keys,
            sorted by relevance (highest score first)
        """
        if top_k is None:
            top_k = self.top_k
        
        # Embed the query
        query_embedding = np.array(self.embed_model.embed_query(query))
        
        # Compute cosine similarity with all page embeddings
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarities
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            results.append({
                'title': self.pages[idx]['title'],
                'subtitle': self.pages[idx]['subtitle'],
                'url': self.pages[idx]['url'],
                'score': float(similarities[idx])
            })
        
        return results
    
    def format_results(self, results: List[Dict[str, str]], include_scores: bool = False) -> str:
        """
        Format results as a readable string.
        
        Args:
            results: List of page results from find_relevant_pages()
            include_scores: Whether to include similarity scores
        
        Returns:
            Formatted string with page recommendations
        """
        if not results:
            return "No relevant pages found."
        
        lines = []
        for i, page in enumerate(results, 1):
            title = page['title']
            subtitle = page.get('subtitle', '')
            url = page['url']
            score = page.get('score', 0)
            
            # Format title with subtitle if available
            if subtitle and subtitle != title:
                display_title = f"{title} - {subtitle}"
            else:
                display_title = title
            
            line = f"{i}. {display_title}\n   {url}"
            if include_scores:
                line += f" (score: {score:.3f})"
            lines.append(line)
        
        return "\n".join(lines)


def find_pages(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, str]]:
    """
    Convenience function to find relevant pages for a query.
    
    Args:
        query: User query text
        top_k: Number of top pages to return
    
    Returns:
        List of relevant pages
    """
    finder = SmartPageFinder()
    return finder.find_relevant_pages(query, top_k=top_k)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Page Finder - Find relevant pages for a query")
    parser.add_argument("query", type=str, help="User query")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of results to return")
    parser.add_argument("--scores", action="store_true", help="Include similarity scores in output")
    
    args = parser.parse_args()
    
    finder = SmartPageFinder()
    results = finder.find_relevant_pages(args.query, top_k=args.top_k)
    
    print(f"\nRelevant pages for query: '{args.query}'\n")
    print(finder.format_results(results, include_scores=args.scores))

