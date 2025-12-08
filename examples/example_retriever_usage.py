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
    
    # Query - namespace is automatically detected from the query
    query = "איך משלמים ארנונה?"
    results = retriever.retrieve(
        query=query, 
        top_k=5,
        strategy=None,  # Can use "baseline", "sentence", or "adaptive" to filter
    )
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i} (score: {result['score']:.4f}):")
        print(f"  Namespace: {result.get('namespace', 'unknown')}")
        print(f"  Strategy: {result.get('chunking_strategy', 'unknown')}")
        print(f"  Title: {result.get('title', 'N/A')}")
        print(f"  URL: {result.get('url', 'N/A')}")
        content_preview = result.get('chunk_text_only', result.get('text', ''))
        print(f"  Content preview: {content_preview[:200]}...")
        print()


def example_with_strategy():
    """Example with chunking strategy filter."""
    print("="*60)
    print("EXAMPLE 2: Retrieval with Strategy Filter")
    print("="*60)
    
    retriever = Retriever(
        api_keys_path="utils/api_keys.json",
        index_name="haifa-municipality-rag-index",
    )
    
    query = "מוקדי שירות"
    # Filter by chunking strategy - only get results from "adaptive" strategy
    results = retriever.retrieve(query=query, top_k=2, strategy="adaptive")
    
    print(f"\nQuery: {query}")
    print(f"Strategy filter: adaptive")
    print(f"Found {len(results)} results\n")


def example_with_strategy_filter():
    """Example with different chunking strategies."""
    print("="*60)
    print("EXAMPLE 3: Comparing Different Strategies")
    print("="*60)
    
    retriever = Retriever(
        api_keys_path="utils/api_keys.json",
        index_name="haifa-municipality-rag-index",
    )
    
    query = "ארנונה"
    
    # Try different strategies
    for strategy in ["baseline", "sentence", "adaptive"]:
        results = retriever.retrieve(query=query, top_k=3, strategy=strategy)
        print(f"\nStrategy: {strategy}")
        print(f"Found {len(results)} results")
        if results:
            print(f"  Best score: {results[0]['score']:.4f}")


def example_batch_retrieval():
    """Example of batch retrieval for multiple queries."""
    print("="*60)
    print("EXAMPLE 4: Multiple Queries")
    print("="*60)
    
    retriever = Retriever(
        api_keys_path="./utils/api_keys.json",
        index_name="haifa-municipality-rag-index",
    )
    
    queries = [
        "איך משלמים ארנונה?",
        "מוקדי שירות",
        "זימון תורים",
    ]
    
    # Note: retrieve_batch() doesn't exist, so we call retrieve() in a loop
    for query in queries:
        results = retriever.retrieve(query=query, top_k=2)
        print(f"\nQuery: {query}")
        print(f"Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.get('title', 'N/A')} (score: {result['score']:.4f})")


def example_namespace_detection():
    """Example showing automatic namespace detection."""
    print("="*60)
    print("EXAMPLE 5: Automatic Namespace Detection")
    print("="*60)
    
    retriever = Retriever(
        api_keys_path="utils/api_keys.json",
        index_name="haifa-municipality-rag-index",
    )
    
    # Different queries will automatically map to different namespaces
    queries = [
        "איך משלמים ארנונה?",  # Should map to "arnona"
        "מה המחיר של חניה?",    # Should map to "parking"
        "איך מקבלים היתר בנייה?",  # Should map to "engineering"
    ]
    
    for query in queries:
        results = retriever.retrieve(query=query, top_k=1)
        if results:
            detected_ns = results[0].get('namespace', 'unknown')
            print(f"\nQuery: {query}")
            print(f"Detected namespace: {detected_ns}")
        else:
            print(f"\nQuery: {query}")
            print("No results found")


if __name__ == "__main__":
    # Uncomment the examples you want to run
    
    example_basic_retrieval()
    # example_with_strategy()
    # example_with_strategy_filter()
    # example_batch_retrieval()
    # example_namespace_detection()

