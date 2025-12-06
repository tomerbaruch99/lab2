"""
Compare different chunking configurations for retrieval quality.

This script evaluates retrieval performance across different chunk size/overlap
configurations using a labeled test set. It filters results by chunk_config
metadata to compare configurations fairly.

Usage:
    python utils/compare_chunk_configs.py \
        --testset tests/embedding_testset.json \
        --chunk_configs chunk500_overlap100,chunk1000_overlap200,chunk1500_overlap300
"""

import sys
import os
import json
import argparse
from typing import List, Dict, Any

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from retriever import Retriever
from utils.config import DEFAULT_INDEX_NAME, DEFAULT_API_KEYS_PATH, DEFAULT_EMBEDDING_MODEL


def load_testset(path: str) -> List[Dict[str, Any]]:
    """Load test set from JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Testset file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Normalize: data can be {"queries": [...]} or just [...]
    if isinstance(data, dict) and "queries" in data:
        queries = data["queries"]
    else:
        queries = data
    
    if not isinstance(queries, list):
        raise ValueError("Testset must be a list of queries or an object with 'queries' list")
    
    return queries


def is_doc_relevant(doc: Dict[str, Any]) -> bool:
    """Determine if a document is labeled as relevant."""
    if "is_relevant" in doc:
        return bool(doc["is_relevant"])
    
    label = str(doc.get("label", "")).strip().lower()
    relevant_labels = {"relevant", "pos", "positive", "gold", "true"}
    return label in relevant_labels


def evaluate_chunk_config(
    chunk_config: str,
    test_queries: List[Dict[str, Any]],
    retriever: Retriever,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Evaluate a specific chunk configuration.
    
    Args:
        chunk_config: Chunk configuration string (e.g., "chunk1000_overlap200")
        test_queries: List of test queries with labeled documents
        retriever: Retriever instance
        top_k: Top-k for hit rate calculation
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {chunk_config}")
    print(f"{'='*60}")
    
    # Filter by chunk_config
    filter_dict = {"chunk_config": chunk_config}
    
    num_queries = 0
    top1_hits = 0
    topk_hits = 0
    total_relevant_retrieved = 0
    total_relevant_available = 0
    
    for query_data in test_queries:
        query_text = query_data["query"]
        labeled_docs = query_data["documents"]
        
        # Get relevant document texts for this query
        relevant_texts = {
            doc["text"] for doc in labeled_docs if is_doc_relevant(doc)
        }
        
        if not relevant_texts:
            continue  # Skip queries with no relevant docs
        
        num_queries += 1
        total_relevant_available += len(relevant_texts)
        
        try:
            # Retrieve with strategy filter (chunk_config is now chunking_strategy)
            # Note: filter_dict and file type filtering are no longer supported
            results = retriever.retrieve(
                query=query_text,
                top_k=top_k,
                strategy=None,  # Can filter by "baseline", "sentence", or "adaptive"
                include_metadata=True,
            )
            
            # Check if relevant documents were retrieved
            retrieved_texts = {
                r.get("chunk_text_only", r.get("text", ""))
                for r in results
            }
            
            # Count relevant documents retrieved
            relevant_retrieved = sum(
                1 for text in retrieved_texts
                if any(rel_text in text or text in rel_text for rel_text in relevant_texts)
            )
            total_relevant_retrieved += relevant_retrieved
            
            # Top-1 hit
            if results and relevant_retrieved > 0:
                # Check if first result is relevant
                first_result_text = results[0].get("chunk_text_only", results[0].get("text", ""))
                if any(rel_text in first_result_text or first_result_text in rel_text 
                       for rel_text in relevant_texts):
                    top1_hits += 1
            
            # Top-k hit
            if relevant_retrieved > 0:
                topk_hits += 1
                
        except Exception as e:
            print(f"   ⚠️ Error processing query '{query_text[:50]}...': {e}")
            continue
    
    if num_queries == 0:
        return {
            "chunk_config": chunk_config,
            "success": False,
            "error": "No queries with relevant documents",
        }
    
    # Calculate metrics
    top1_hit_rate = top1_hits / num_queries
    topk_hit_rate = topk_hits / num_queries
    recall = total_relevant_retrieved / total_relevant_available if total_relevant_available > 0 else 0.0
    
    return {
        "chunk_config": chunk_config,
        "success": True,
        "num_queries": num_queries,
        "top1_hit_rate": top1_hit_rate,
        "topk_hit_rate": topk_hit_rate,
        "recall": recall,
        "total_relevant_available": total_relevant_available,
        "total_relevant_retrieved": total_relevant_retrieved,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare chunking configurations for retrieval quality"
    )
    parser.add_argument(
        "--testset",
        type=str,
        required=True,
        help="Path to JSON test set file"
    )
    parser.add_argument(
        "--chunk_configs",
        type=str,
        required=True,
        help="Comma-separated list of chunk configs (e.g., 'chunk500_overlap100,chunk1000_overlap200')"
    )
    parser.add_argument(
        "--api_keys_path",
        type=str,
        default=DEFAULT_API_KEYS_PATH,
        help="Path to API keys file"
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default=DEFAULT_INDEX_NAME,
        help="Pinecone index name"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top-k for hit rate calculation"
    )
    
    args = parser.parse_args()
    
    # Parse chunk configs
    chunk_configs = [c.strip() for c in args.chunk_configs.split(",")]
    
    # Load test set
    print(f"[STEP] Loading test set from {args.testset}...")
    try:
        test_queries = load_testset(args.testset)
    except Exception as e:
        print(f"❌ Failed to load test set: {e}")
        sys.exit(1)
    
    print(f"[OK] Loaded {len(test_queries)} test queries")
    
    # Initialize retriever
    print(f"\n[STEP] Initializing retriever...")
    retriever = Retriever(
        api_keys_path=args.api_keys_path,
        embedding_model_name=args.embedding_model,
        index_name=args.index_name,
    )
    print(f"[OK] Retriever ready")
    
    # Evaluate each chunk config
    print(f"\n{'='*60}")
    print(f"EVALUATING {len(chunk_configs)} CHUNK CONFIGURATIONS")
    print(f"{'='*60}")
    
    results = []
    for chunk_config in chunk_configs:
        result = evaluate_chunk_config(
            chunk_config=chunk_config,
            test_queries=test_queries,
            retriever=retriever,
            top_k=args.top_k,
        )
        results.append(result)
        
        if result["success"]:
            print(f"\n✅ Results for {chunk_config}:")
            print(f"   Top-1 hit rate: {result['top1_hit_rate']:.3f}")
            print(f"   Top-{args.top_k} hit rate: {result['topk_hit_rate']:.3f}")
            print(f"   Recall: {result['recall']:.3f}")
            print(f"   Relevant retrieved: {result['total_relevant_retrieved']}/{result['total_relevant_available']}")
        else:
            print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Summary comparison
    successful = [r for r in results if r["success"]]
    if successful:
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        
        # Sort by top-1 hit rate
        successful.sort(key=lambda x: x["top1_hit_rate"], reverse=True)
        
        print(f"\nRanked by Top-1 Hit Rate:\n")
        for i, result in enumerate(successful, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"{medal} {result['chunk_config']}")
            print(f"   Top-1: {result['top1_hit_rate']:.3f} | "
                  f"Top-{args.top_k}: {result['topk_hit_rate']:.3f} | "
                  f"Recall: {result['recall']:.3f}")
        
        # Also sort by recall
        successful_by_recall = sorted(successful, key=lambda x: x["recall"], reverse=True)
        print(f"\nRanked by Recall:\n")
        for i, result in enumerate(successful_by_recall, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"{medal} {result['chunk_config']}")
            print(f"   Recall: {result['recall']:.3f} | "
                  f"Top-1: {result['top1_hit_rate']:.3f} | "
                  f"Top-{args.top_k}: {result['topk_hit_rate']:.3f}")


if __name__ == "__main__":
    main()

