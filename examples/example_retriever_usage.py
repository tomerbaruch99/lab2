"""
Example usage of the RAG Retriever

This script demonstrates how to use the retriever for various scenarios.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever import Retriever


def example_basic_retrieval():
    """Basic retrieval example."""
    print("="*60)
    print("EXAMPLE 1: Basic Retrieval")
    print("="*60)
    
    # Initialize retriever
    # Using the default embedding model optimized for Hebrew: paraphrase-multilingual-MiniLM-L12-v2
    retriever = Retriever(
        api_keys_path="utils/api_keys.json",
        embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        index_name="haifa-municipality-rag-index",
    )
    
    # Query - exclude PDFs to get better HTML/TXT results
    # PDFs often have generic titles and may not be as relevant
    query = "איך משלמים ארנונה?"
    results = retriever.retrieve(
        query, 
        top_k=5,  # Get more results initially for better filtering
        exclude_file_types=["pdf"],  # Exclude PDFs to get clearer HTML/TXT results
        prefer_txt_html=True  # Prefer HTML/TXT over PDFs when available
    )
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results (PDFs excluded):\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i} (score: {result['score']:.4f}):")
        print(f"  File type: {result.get('file_type', 'unknown')}")
        print(f"  Title: {result.get('title', 'N/A')}")
        print(f"  URL: {result.get('url', 'N/A')}")
        content_preview = result.get('chunk_text_only', result.get('text', ''))
        print(f"  Content preview: {content_preview[:200]}...")
        print()


def example_with_namespace():
    """Example with namespace (e.g., dev environment)."""
    print("="*60)
    print("EXAMPLE 2: Retrieval with Namespace")
    print("="*60)
    
    retriever = Retriever(
        api_keys_path="utils/api_keys.json",
        index_name="haifa-municipality-rag-index",
        namespace="dev",  # Use dev namespace
    )
    
    query = "מוקדי שירות"
    results = retriever.retrieve(query, top_k=2)
    
    print(f"\nQuery: {query}")
    print(f"Namespace: dev")
    print(f"Found {len(results)} results\n")


def example_with_filter():
    """Example with metadata filter."""
    print("="*60)
    print("EXAMPLE 3: Retrieval with Filter")
    print("="*60)
    
    retriever = Retriever(
        api_keys_path="utils/api_keys.json",
        index_name="haifa-municipality-rag-index",
    )
    
    # Filter by specific document
    filter_dict = {"doc_id": "resident-service"}
    
    query = "ארנונה"
    results = retriever.retrieve(query, top_k=3, filter_dict=filter_dict)
    
    print(f"\nQuery: {query}")
    print(f"Filter: doc_id = 'resident-service'")
    print(f"Found {len(results)} results\n")


def example_batch_retrieval():
    """Example of batch retrieval for multiple queries."""
    print("="*60)
    print("EXAMPLE 4: Batch Retrieval")
    print("="*60)
    
    retriever = Retriever(
        api_keys_path="../utils/api_keys.json",
        index_name="haifa-municipality-rag-index",
    )
    
    queries = [
        "איך משלמים ארנונה?",
        "מוקדי שירות",
        "זימון תורים",
    ]
    
    all_results = retriever.retrieve_batch(queries, top_k=2)
    
    for query, results in zip(queries, all_results):
        print(f"\nQuery: {query}")
        print(f"Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.get('title', 'N/A')} (score: {result['score']:.4f})")


def example_delete_document():
    """Example of deleting a document for reindexing."""
    print("="*60)
    print("EXAMPLE 5: Delete Document for Reindexing")
    print("="*60)
    
    retriever = Retriever(
        api_keys_path="utils/api_keys.json",
        index_name="haifa-municipality-rag-index",
    )
    
    # Delete all chunks for a specific document
    doc_id = "resident-service"
    print(f"\nDeleting all chunks for doc_id: {doc_id}")
    retriever.delete_by_doc_id(doc_id)
    print("Done! You can now reindex this document.")


if __name__ == "__main__":
    # Uncomment the examples you want to run
    
    example_basic_retrieval()
    # example_with_namespace()
    # example_with_filter()
    # example_batch_retrieval()
    # example_delete_document()

