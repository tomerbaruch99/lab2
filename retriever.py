"""
Haifa Municipality RAG Retriever
================================
Retrieves top-K relevant chunks from Pinecone for a given user question.

This module:
1. Embeds user questions using the same model as indexing
2. Queries Pinecone for similar chunks
3. Returns ranked results with metadata

Usage as script:
    python retriever.py \
        --query "איך משלמים ארנונה?" \
        --top_k 5 \
        --index_name haifa-municipality-rag-index

Usage as module:
    from retriever import Retriever
    
    retriever = Retriever(
        api_keys_path="api_keys.json",
        embedding_model_name="all-MiniLM-L6-v2",
        index_name="haifa-municipality-rag-index"
    )
    
    results = retriever.retrieve("איך משלמים ארנונה?", top_k=5)
"""

import argparse
from typing import List, Dict, Optional, Any

from pinecone import Pinecone

from utils import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_NAME,
    DEFAULT_API_KEYS_PATH,
    DEFAULT_TOP_K,
    load_pinecone_api_key,
    EmbeddingModel,
)


# --- Retriever class ---

class Retriever:
    """
    RAG Retriever for Haifa Municipality data.
    
    Retrieves relevant chunks from Pinecone based on semantic similarity.
    """
    
    def __init__(
        self,
        api_keys_path: str = DEFAULT_API_KEYS_PATH,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        index_name: str = DEFAULT_INDEX_NAME,
        namespace: Optional[str] = None,
    ):
        """
        Initialize the retriever.
        
        Args:
            api_keys_path: Path to API keys JSON file
            embedding_model_name: Name of the SentenceTransformer model
            index_name: Name of the Pinecone index
            namespace: Optional namespace (for dev/prod/language separation)
        """
        # Initialize Pinecone
        pinecone_api_key = load_pinecone_api_key(api_keys_path)
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        self.index_name = index_name
        self.namespace = namespace
        
        # Initialize embedding model
        self.embed_model = EmbeddingModel(embedding_model_name)
        
        print(f"[INFO] Retriever initialized")
        print(f"[INFO] Index: {index_name}")
        if namespace:
            print(f"[INFO] Namespace: {namespace}")
        print(f"[INFO] Embedding model: {embedding_model_name}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filter_dict: Optional[Dict[str, Any]] = None,
        exclude_file_types: Optional[List[str]] = None,
        include_file_types: Optional[List[str]] = None,
        include_metadata: bool = True,
        include_values: bool = False,
        prefer_txt_html: bool = True,
    ) -> List[Dict]:
        """
        Retrieve top-K relevant chunks for a query.
        
        Args:
            query: User question/query string
            top_k: Number of top results to return
            filter_dict: Optional metadata filter (e.g., {"doc_id": "specific_doc"})
            exclude_file_types: List of file types to exclude (e.g., ["pdf"])
            include_file_types: List of file types to include (e.g., ["html", "txt"])
            include_metadata: Whether to include metadata in results
            include_values: Whether to include vector values in results
            prefer_txt_html: If True, prefer txt/html over PDF unless there aren't enough txt/html results
        
        Returns:
            List of dictionaries with:
                - id: Chunk ID (e.g., "doc_id::chunk-0")
                - score: Similarity score
                - metadata: Dict with text, chunk_text_only, url, title, subtitle, file_type, etc.
        
        Note: If both exclude_file_types and include_file_types are provided, exclude takes precedence.
        """
        # Embed the query
        query_embedding = self.embed_model.embed_query(query)
        
        # Prepare query parameters - request more results if we need to filter by file type
        # or if we want to re-rank by file type preference
        # This ensures we get enough non-filtered results for re-ranking
        needs_extra_results = (exclude_file_types or include_file_types or prefer_txt_html)
        query_k = top_k * 3 if needs_extra_results else top_k
        
        query_params = {
            "vector": query_embedding,
            "top_k": query_k,
            "include_metadata": include_metadata,
            "include_values": include_values,
        }
        
        # Add filter if provided
        if filter_dict:
            query_params["filter"] = filter_dict
        
        # Query Pinecone
        if self.namespace:
            results = self.index.query(**query_params, namespace=self.namespace)
        else:
            results = self.index.query(**query_params)
        
        # Format results and apply file type filtering
        formatted_results = []
        for match in results.matches:
            # Get file type from metadata
            file_type = None
            if include_metadata and match.metadata:
                file_type = match.metadata.get("file_type", "html")
            
            # Apply file type filtering
            if exclude_file_types and file_type in exclude_file_types:
                continue
            if include_file_types and file_type not in include_file_types:
                continue
            
            result = {
                "id": match.id,
                "score": match.score,
            }
            
            if include_metadata and match.metadata:
                result["metadata"] = match.metadata
                # Extract commonly used fields for convenience
                result["text"] = match.metadata.get("text", "")
                result["chunk_text_only"] = match.metadata.get("chunk_text_only", "")
                result["url"] = match.metadata.get("url", "")
                result["title"] = match.metadata.get("title", "")
                result["subtitle"] = match.metadata.get("subtitle", "")
                result["doc_id"] = match.metadata.get("doc_id", "")
                result["chunk_id"] = match.metadata.get("chunk_id", "")
                result["file_type"] = file_type
            
            if include_values and match.values:
                result["values"] = match.values
            
            formatted_results.append(result)
        
        # Apply file type preference re-ranking if enabled
        if prefer_txt_html and not exclude_file_types and not include_file_types:
            formatted_results = self._reorder_by_file_type_preference(formatted_results, top_k)
        else:
            # Just take top_k if no re-ranking needed
            formatted_results = formatted_results[:top_k]
        
        return formatted_results
    
    def _reorder_by_file_type_preference(self, results: List[Dict], top_k: int) -> List[Dict]:
        """
        Re-order results to prefer txt/html over PDF, but still include PDFs if needed.
        
        Prioritizes txt/html files, but includes PDFs to fill top_k if there aren't
        enough txt/html results to satisfy the request.
        
        Args:
            results: List of result dictionaries with file_type field
            top_k: Target number of results to return
        
        Returns:
            Re-ordered list with txt/html first, then PDFs, limited to top_k
        """
        preferred_types = {"html", "txt"}
        
        # Separate results by file type preference
        preferred_results = []
        pdf_results = []
        other_results = []
        
        for result in results:
            file_type = result.get("file_type", "html")
            if file_type in preferred_types:
                preferred_results.append(result)
            elif file_type == "pdf":
                pdf_results.append(result)
            else:
                other_results.append(result)
        
        # Combine: preferred first, then PDFs, then others, up to top_k
        reordered = preferred_results + pdf_results + other_results
        return reordered[:top_k]
    
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = DEFAULT_TOP_K,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict]]:
        """
        Retrieve results for multiple queries in batch.
        
        Args:
            queries: List of query strings
            top_k: Number of top results per query
            filter_dict: Optional metadata filter
        
        Returns:
            List of result lists, one per query
        """
        # Embed all queries
        query_embeddings = self.embed_model.embed(queries, show_progress=True)
        
        # Query Pinecone for each embedding
        all_results = []
        for query_embedding in query_embeddings:
            query_params = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True,
            }
            
            if filter_dict:
                query_params["filter"] = filter_dict
            
            if self.namespace:
                results = self.index.query(**query_params, namespace=self.namespace)
            else:
                results = self.index.query(**query_params)
            
            # Format results
            formatted_results = []
            for match in results.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata or {},
                    "text": match.metadata.get("text", "") if match.metadata else "",
                    "chunk_text_only": match.metadata.get("chunk_text_only", "") if match.metadata else "",
                    "url": match.metadata.get("url", "") if match.metadata else "",
                    "title": match.metadata.get("title", "") if match.metadata else "",
                    "subtitle": match.metadata.get("subtitle", "") if match.metadata else "",
                    "file_type": match.metadata.get("file_type", "html") if match.metadata else "html",
                }
                formatted_results.append(result)
            
            all_results.append(formatted_results)
        
        return all_results
    
    def delete_by_doc_id(self, doc_id: str) -> None:
        """
        Delete all chunks for a specific document.
        
        Useful for reindexing a specific document.
        
        Args:
            doc_id: Document ID to delete
        """
        filter_dict = {"doc_id": doc_id}
        if self.namespace:
            self.index.delete(filter=filter_dict, namespace=self.namespace)
        else:
            self.index.delete(filter=filter_dict)
        print(f"[INFO] Deleted chunks for doc_id: {doc_id}")


# --- CLI ---

def format_result(result: Dict, show_full_text: bool = False) -> str:
    """Format a single result for display."""
    lines = []
    lines.append(f"Score: {result['score']:.4f}")
    lines.append(f"ID: {result['id']}")
    
    if result.get("title"):
        lines.append(f"Title: {result['title']}")
    if result.get("subtitle"):
        lines.append(f"Subtitle: {result['subtitle']}")
    if result.get("url"):
        lines.append(f"URL: {result['url']}")
    if result.get("doc_id"):
        lines.append(f"Doc ID: {result['doc_id']}")
    if result.get("chunk_id") is not None:
        lines.append(f"Chunk ID: {result['chunk_id']}")
    
    lines.append("")
    if show_full_text:
        text = result.get("text", "")
    else:
        text = result.get("chunk_text_only", result.get("text", ""))
    
    # Truncate long text
    if len(text) > 500:
        text = text[:500] + "..."
    lines.append(f"Content: {text}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve relevant chunks from Haifa Municipality RAG index"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User question/query"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top results to return (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--api_keys_path",
        type=str,
        default=DEFAULT_API_KEYS_PATH,
        help="Path to API keys JSON file"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name"
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default=DEFAULT_INDEX_NAME,
        help="Pinecone index name"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Optional namespace (e.g., 'dev', 'prod', 'hebrew', 'arabic')"
    )
    parser.add_argument(
        "--show_full_text",
        action="store_true",
        help="Show full text (with title/subtitle) instead of chunk_text_only"
    )
    parser.add_argument(
        "--filter_doc_id",
        type=str,
        default=None,
        help="Filter results by specific doc_id"
    )
    parser.add_argument(
        "--exclude_file_types",
        type=str,
        default=None,
        help="Comma-separated file types to exclude (e.g., 'pdf' or 'pdf,doc')"
    )
    parser.add_argument(
        "--include_file_types",
        type=str,
        default=None,
        help="Comma-separated file types to include (e.g., 'html,txt')"
    )
    
    args = parser.parse_args()
    
    # Initialize retriever
    print("[STEP] Initializing retriever...")
    retriever = Retriever(
        api_keys_path=args.api_keys_path,
        embedding_model_name=args.embedding_model,
        index_name=args.index_name,
        namespace=args.namespace,
    )
    
    # Prepare filter if needed
    filter_dict = None
    if args.filter_doc_id:
        filter_dict = {"doc_id": args.filter_doc_id}
    
    # Parse file type filters
    exclude_file_types = None
    if args.exclude_file_types:
        exclude_file_types = [ft.strip() for ft in args.exclude_file_types.split(",")]
    
    include_file_types = None
    if args.include_file_types:
        include_file_types = [ft.strip() for ft in args.include_file_types.split(",")]
    
    # Retrieve
    print(f"\n[STEP] Retrieving top-{args.top_k} results for query...")
    print(f"[QUERY] {args.query}\n")
    if exclude_file_types:
        print(f"[FILTER] Excluding file types: {exclude_file_types}")
    if include_file_types:
        print(f"[FILTER] Including only file types: {include_file_types}")
    
    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        filter_dict=filter_dict,
        exclude_file_types=exclude_file_types,
        include_file_types=include_file_types,
    )
    
    # Display results
    print("="*60)
    print(f"RETRIEVED {len(results)} RESULTS")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(format_result(result, show_full_text=args.show_full_text))
        print("-" * 60)
    
    if not results:
        print("[INFO] No results found. Try:")
        print("  - Adjusting the query")
        print("  - Checking if the index is populated")
        print("  - Verifying namespace matches (if used)")


if __name__ == "__main__":
    main()

