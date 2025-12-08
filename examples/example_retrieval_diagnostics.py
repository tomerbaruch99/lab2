"""
Diagnostic example to understand why retrieval results might be poor.

This script helps identify common issues with retrieval quality.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever import Retriever


def diagnose_retrieval_issues():
    """Diagnose common retrieval issues."""
    print("="*60)
    print("RETRIEVAL DIAGNOSTICS")
    print("="*60)
    
    query = "איך משלמים ארנונה?"
    
    retriever = Retriever(
        api_keys_path="./utils/api_keys.json",
        embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        index_name="haifa-municipality-rag-index",
    )
    
    print(f"\nQuery: {query}\n")
    
    # Test 1: All results (including PDFs)
    print("="*60)
    print("TEST 1: All file types (including PDFs)")
    print("="*60)
    all_results = retriever.retrieve(query, top_k=10)
    print(f"Found {len(all_results)} results\n")
    
    pdf_count = sum(1 for r in all_results if r.get("file_type") == "pdf")
    html_count = sum(1 for r in all_results if r.get("file_type") in ["html", "txt"])
    print(f"PDF results: {pdf_count}")
    print(f"HTML/TXT results: {html_count}")
    print(f"Other: {len(all_results) - pdf_count - html_count}")
    
    # Show strategy distribution
    print("\nChunking Strategy Distribution:")
    strategy_counts = {}
    for r in all_results:
        strategy = r.get("chunking_strategy", "unknown")
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    for strategy, count in strategy_counts.items():
        print(f"  - {strategy}: {count} results")
    
    # Show namespace distribution
    print("\nNamespace Distribution:")
    namespace_counts = {}
    for r in all_results:
        ns = r.get("namespace", "unknown")
        namespace_counts[ns] = namespace_counts.get(ns, 0) + 1
    for ns, count in namespace_counts.items():
        print(f"  - {ns}: {count} results")
    
    # Show doc_type distribution
    print("\nDocument Type Distribution:")
    doc_type_counts = {}
    for r in all_results:
        doc_type = r.get("metadata", {}).get("doc_type", "unknown") if isinstance(r.get("metadata"), dict) else "unknown"
        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
    for doc_type, count in doc_type_counts.items():
        print(f"  - {doc_type}: {count} results")
    
    # Test 2: Different chunking strategies
    print("\n" + "="*60)
    print("TEST 2: Different Chunking Strategies")
    print("="*60)
    for strategy in ["baseline", "sentence", "adaptive"]:
        strategy_results = retriever.retrieve(query, top_k=5, strategy=strategy)
        print(f"\nStrategy: {strategy}")
        print(f"Found {len(strategy_results)} results")
        if strategy_results:
            print(f"  Best score: {strategy_results[0]['score']:.4f}")
            print(f"  Namespace: {strategy_results[0].get('namespace', 'unknown')}")
    
    # Test 3: Namespace detection
    print("\n" + "="*60)
    print("TEST 3: Namespace Detection")
    print("="*60)
    all_results = retriever.retrieve(query, top_k=10)
    namespace_counts = {}
    for result in all_results:
        ns = result.get('namespace', 'unknown')
        namespace_counts[ns] = namespace_counts.get(ns, 0) + 1
    print("Namespace distribution:")
    for ns, count in namespace_counts.items():
        print(f"  {ns}: {count} results")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("\n1. Use strategy parameter to filter by chunking strategy:")
    print("   Use: strategy='adaptive' or strategy='sentence'")
    print("\n2. Namespace is automatically detected from query:")
    print("   Queries with 'ארנונה' map to 'arnona' namespace")
    print("\n3. Embedding model information:")
    print("   Current: 'paraphrase-multilingual-MiniLM-L12-v2'")
    print("   (Optimized for Hebrew and multilingual content)")
    print("\n4. Check if indexed content actually contains relevant information")
    print("   about your query topic.")


if __name__ == "__main__":
    diagnose_retrieval_issues()

