"""
Haifa RAG Retriever
====================================

Retriever supporting:
- Automatic namespace detection from Hebrew query
- Metadata-aware retrieval (doc_type, chunking_strategy)
- Fallback search when namespace returns no results
- Clean, modern architecture matching the new data preparation pipeline
"""

import argparse
from typing import List, Dict, Optional, Any
from pinecone import Pinecone
import os

from utils import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_NAME,
    DEFAULT_API_KEYS_PATH,
    DEFAULT_TOP_K,
    NAMESPACE_RULES,
    FALLBACK_NAMESPACE,
    load_pinecone_api_key,
    EmbeddingModel,
)


def detect_namespace(query: str) -> str:
    """
    Rule-based classifier that determines the correct namespace for a Hebrew query.
    
    This function uses keyword matching to map user queries to appropriate namespaces
    in the Pinecone index. Namespaces help organize documents by topic (arnona, parking,
    water, etc.) and improve retrieval precision.
    
    Args:
        query: User query string in Hebrew
        
    Returns:
        Namespace string (e.g., "arnona", "parking", "water") or "general" as fallback
        
    Example:
        >>> detect_namespace("איך משלמים ארנונה?")
        'arnona'
        >>> detect_namespace("מה המחיר של חניה?")
        'parking'
        >>> detect_namespace("שאלה כללית")
        'general'
        
    Note:
        The function checks keywords in order of NAMESPACE_RULES dictionary.
        First match wins. If no keywords match, returns FALLBACK_NAMESPACE ("general").
    """
    # Iterate through namespace rules (ordered dictionary)
    # Each namespace has a list of Hebrew keywords that indicate relevance
    for ns, keywords in NAMESPACE_RULES.items():
        for kw in keywords:
            # Simple substring matching (case-sensitive for Hebrew)
            if kw in query:
                return ns
    # Return fallback namespace if no keywords match
    return FALLBACK_NAMESPACE


# ============================================================
# Retriever Class
# ============================================================

class Retriever:
    """
    Updated RAG Retriever for the new Haifa municipality data pipeline.
    """

    def __init__(
        self,
        api_keys_path: str = DEFAULT_API_KEYS_PATH,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        index_name: Optional[str] = None,
    ):
        # Pinecone init
        api_key = load_pinecone_api_key(api_keys_path)
        self.pc = Pinecone(api_key=api_key)
        
        # Determine index name: parameter > environment variable > default
        if index_name is None:
            index_name = os.environ.get("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
        
        self.index = self.pc.Index(index_name)

        # Embedding model (force CPU to avoid CUDA compatibility issues)
        self.embed_model = EmbeddingModel(embedding_model_name, device="cpu")

        print(f"[INFO] Retriever ready | Index: {index_name}")

    # ------------------------------------------------------------

    def _build_metadata_filter(self, strategy: Optional[str]) -> Dict:
        """
        Build metadata filter dictionary for Pinecone query.
        
        Pinecone allows filtering results by metadata fields. This function creates
        a filter that restricts results to a specific chunking strategy if provided.
        
        Args:
            strategy: Optional chunking strategy to filter by
                     ("baseline", "sentence", "adaptive", or None for all)
        
        Returns:
            Dictionary with metadata filters, or empty dict if no filter needed
            
        Note:
            If strategy is None, returns empty dict (no filtering).
            This allows retrieving chunks from all chunking strategies.
        """
        metadata_filter = {}
        if strategy:
            # Filter by chunking strategy (baseline, sentence, or adaptive)
            metadata_filter["chunking_strategy"] = strategy
        return metadata_filter

    def _query_pinecone(
        self,
        vector: List[float],
        top_k: int,
        namespace: str,
        metadata_filter: Optional[Dict],
        include_metadata: bool
    ):
        """
        Query Pinecone index with given parameters.
        
        This is a low-level wrapper around Pinecone's query method that handles
        vector similarity search with optional metadata filtering.
        
        Args:
            vector: Query embedding vector (list of floats)
            top_k: Number of top results to retrieve
            namespace: Namespace to search in (e.g., "arnona", "parking", "general")
            metadata_filter: Optional metadata filter dict (e.g., {"chunking_strategy": "adaptive"})
            include_metadata: Whether to include full metadata in results
            
        Returns:
            Pinecone query response object with matches and scores
            
        Note:
            The query uses cosine similarity by default (configured at index creation).
            Results are sorted by similarity score (highest first).
        """
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=include_metadata,
            namespace=namespace,
            filter=metadata_filter if metadata_filter else None,
        )

    def _format_results(self, matches, include_metadata: bool) -> List[Dict]:
        """
        Format Pinecone query results into standardized dictionary format.
        
        Converts Pinecone match objects into a consistent dictionary structure
        that's easier to work with throughout the RAG pipeline. Extracts all
        relevant metadata fields and provides defaults for missing values.
        
        Args:
            matches: List of Pinecone match objects from query response
            include_metadata: Whether to include full text field (may be large)
            
        Returns:
            List of dictionaries, each containing:
            - id: Vector ID in Pinecone
            - score: Similarity score (0-1 for cosine similarity)
            - metadata: Full metadata dict from Pinecone
            - text: Full chunk text with title/subtitle (if include_metadata=True)
            - chunk_text_only: Just the chunk content without title/subtitle
            - url: Source page URL
            - title: Page title
            - subtitle: Page subtitle
            - doc_id: Document identifier
            - chunk_id: Chunk index within document
            - doc_type: Document type (pdf, html, doc, etc.)
            - namespace: Namespace this chunk belongs to
            - chunking_strategy: Strategy used to create this chunk
            
        Note:
            The 'text' field includes title/subtitle context, while 'chunk_text_only'
            contains just the chunk content. Use 'chunk_text_only' for display
            and 'text' when you need full context.
        """
        formatted = []
        for m in matches:
            formatted.append({
                "id": m.id,
                "score": m.score,
                "metadata": m.metadata,
                # Full text with title/subtitle (only if metadata requested)
                "text": m.metadata.get("text", "") if include_metadata else "",
                # Just the chunk content without title/subtitle prefix
                "chunk_text_only": m.metadata.get("chunk_text_only", ""),
                "url": m.metadata.get("url", ""),
                "title": m.metadata.get("title", ""),
                "subtitle": m.metadata.get("subtitle", ""),
                "doc_id": m.metadata.get("doc_id", ""),
                "chunk_id": m.metadata.get("chunk_id", ""),
                "doc_type": m.metadata.get("doc_type", ""),
                "namespace": m.metadata.get("namespace", ""),
                "chunking_strategy": m.metadata.get("chunking_strategy", ""),
            })
        return formatted

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        strategy: Optional[str] = None,    # "baseline" / "sentence" / "adaptive"
        include_metadata: bool = True,
    ) -> List[Dict]:
        """
        Main retrieval function.
        
        Args:
            query: User query string
            top_k: Number of results to retrieve
            strategy: Optional chunking strategy filter
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of formatted result dictionaries
        """
        # Step 1: Detect namespace from query using keyword matching
        # This helps narrow down the search to relevant document categories
        namespace = detect_namespace(query)
        print(f"[INFO] Query mapped to namespace: {namespace}")

        # Step 2: Embed the query using the same embedding model used for indexing
        # This ensures consistent vector space representation
        q_emb = self.embed_model.embed_query(query)

        # Step 3: Build metadata filter if chunking strategy is specified
        # This allows filtering results by chunking strategy (baseline/sentence/adaptive)
        metadata_filter = self._build_metadata_filter(strategy)

        # Step 4: Query Pinecone index with the embedded query vector
        # Returns top_k most similar chunks from the specified namespace
        results = self._query_pinecone(
            vector=q_emb,
            top_k=top_k,
            namespace=namespace,
            metadata_filter=metadata_filter,
            include_metadata=include_metadata,
        )

        # Step 5: Fallback mechanism - if no results in detected namespace,
        # try searching in the general namespace
        # This ensures we always return some results even if namespace detection fails
        if not results.matches:
            print("[WARN] No results in namespace. Falling back to 'general'")
            results = self._query_pinecone(
                vector=q_emb,
                top_k=top_k,
                namespace=FALLBACK_NAMESPACE,
                metadata_filter=metadata_filter,
                include_metadata=include_metadata,
            )

        # Step 6: Format results into standardized dictionary structure
        # This makes the results easier to work with in downstream components
        return self._format_results(results.matches, include_metadata)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve relevant chunks from Pinecone index"
    )
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--strategy", required=False, type=str)
    parser.add_argument("--top_k", default=DEFAULT_TOP_K, type=int)
    args = parser.parse_args()

    retriever = Retriever()
    results = retriever.retrieve(args.query, top_k=args.top_k, strategy=args.strategy)

    print("\nRESULTS:\n")
    for i, r in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Score: {r['score']:.4f}")
        print(f"Namespace: {r['namespace']}")
        print(f"Strategy: {r['chunking_strategy']}")
        print(f"Title: {r['title']}")
        print(f"URL: {r['url']}")
        print(f"Content: {r['chunk_text_only'][:400]}...\n")


if __name__ == "__main__":
    main()
