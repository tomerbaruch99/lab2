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
        api_keys_path="../utils/api_keys.json",
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
    
    # Show PDF issues
    print("\nPDF Issues:")
    pdf_results = [r for r in all_results if r.get("file_type") == "pdf"]
    generic_title_count = sum(1 for r in pdf_results if r.get("title", "").lower() == "pdf document")
    print(f"  - PDFs with generic 'PDF Document' title: {generic_title_count}/{len(pdf_results)}")
    if pdf_results:
        avg_pdf_score = sum(r["score"] for r in pdf_results) / len(pdf_results)
        print(f"  - Average PDF score: {avg_pdf_score:.4f}")
    
    # Show HTML/TXT results
    print("\nHTML/TXT Results:")
    html_results = [r for r in all_results if r.get("file_type") in ["html", "txt"]]
    if html_results:
        avg_html_score = sum(r["score"] for r in html_results) / len(html_results)
        print(f"  - Average HTML/TXT score: {avg_html_score:.4f}")
        print(f"  - Best HTML/TXT title: {html_results[0].get('title', 'N/A')}")
    else:
        print("  - No HTML/TXT results found!")
    
    # Test 2: Exclude PDFs
    print("\n" + "="*60)
    print("TEST 2: Excluding PDFs")
    print("="*60)
    html_only_results = retriever.retrieve(query, top_k=5, exclude_file_types=["pdf"])
    print(f"Found {len(html_only_results)} HTML/TXT results:\n")
    
    for i, result in enumerate(html_only_results[:3], 1):
        print(f"Result {i} (score: {result['score']:.4f}):")
        print(f"  File type: {result.get('file_type')}")
        print(f"  Title: {result.get('title', 'N/A')}")
        print(f"  URL: {result.get('url', 'N/A')[:80]}...")
        print()
    
    # Test 3: With file type preference
    print("="*60)
    print("TEST 3: With file type preference (txt/html preferred)")
    print("="*60)
    preferred_results = retriever.retrieve(query, top_k=5, prefer_txt_html=True)
    print(f"Found {len(preferred_results)} results:\n")
    
    for i, result in enumerate(preferred_results[:3], 1):
        print(f"Result {i} (score: {result['score']:.4f}):")
        print(f"  File type: {result.get('file_type')}")
        print(f"  Title: {result.get('title', 'N/A')}")
        print()
    
    # Recommendations
    print("="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("\n1. EXCLUDE PDFs when they have generic titles:")
    print("   Use: exclude_file_types=['pdf']")
    print("\n2. PREFER HTML/TXT for clearer answers:")
    print("   Use: prefer_txt_html=True")
    print("\n3. Embedding model information:")
    print("   Current: 'paraphrase-multilingual-MiniLM-L12-v2'")
    print("   (Optimized for Hebrew and multilingual content)")
    print("\n4. Check if indexed content actually contains relevant information")
    print("   about your query topic.")


if __name__ == "__main__":
    diagnose_retrieval_issues()

