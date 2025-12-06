"""
Chunking Strategy Evaluation Script
===================================

This script runs the evaluation of different chunking strategies:
- baseline
- sentence
- adaptive

It tests queries across all strategies and saves raw results to CSV files.
For visualization and analysis, use the evaluate_chunking_strategies.ipynb notebook.

Usage:
    python evaluate_chunking_strategies.py \
        --queries_file evaluation_queries.json \
        --output_dir ./evaluation_results \
        --top_k 5
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from retriever import Retriever
from utils import DEFAULT_API_KEYS_PATH, DEFAULT_TOP_K


# ============================================================
# Default Evaluation Queries
# ============================================================

DEFAULT_QUERIES = [
    {
        "query": "איך משלמים ארנונה?",
        "expected_namespace": "arnona",
        "category": "payment",
    },
    {
        "query": "מה המחיר של חניה?",
        "expected_namespace": "parking",
        "category": "pricing",
    },
    {
        "query": "איך מקבלים היתר בנייה?",
        "expected_namespace": "engineering",
        "category": "procedure",
    },
    {
        "query": "מה יש לעשות במקרה של נזילת מים?",
        "expected_namespace": "water",
        "category": "procedure",
    },
    {
        "query": "איך מתקנים חשבון מים?",
        "expected_namespace": "water",
        "category": "procedure",
    },
    {
        "query": "מתי מפנים את האשפה?",
        "expected_namespace": "sanitation",
        "category": "schedule",
    },
    {
        "query": "אילו אירועים יש השבוע?",
        "expected_namespace": "culture",
        "category": "event",
    },
    {
        "query": "איך מתבקש סיוע מהרווחה?",
        "expected_namespace": "welfare",
        "category": "procedure",
    },
    {
        "query": "מה לעשות במקרה של אזעקה?",
        "expected_namespace": "emergency",
        "category": "procedure",
    },
    {
        "query": "מה התעריפים של השירותים העירוניים?",
        "expected_namespace": "general",
        "category": "pricing",
    },
]


# ============================================================
# Metrics Collection
# ============================================================

def evaluate_retrieval(
    retriever: Retriever,
    query: str,
    strategy: str,
    top_k: int = 5,
    expected_namespace: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate retrieval for a single query and strategy.
    
    Returns:
        Dictionary with metrics:
        - avg_score: Average similarity score of retrieved chunks
        - max_score: Highest similarity score
        - min_score: Lowest similarity score
        - num_results: Number of chunks retrieved
        - detected_namespace: Namespace detected by retriever
        - namespace_correct: Whether namespace detection was correct
        - doc_types: Distribution of doc_types in results
        - unique_docs: Number of unique documents
    """
    
    # Suppress print statements during retrieval
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        chunks = retriever.retrieve(
            query=query,
            top_k=top_k,
            strategy=strategy,
            include_metadata=True,
        )
    finally:
        sys.stdout = old_stdout
    
    if not chunks:
        return {
            "avg_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "std_score": 0.0,
            "num_results": 0,
            "detected_namespace": "none",
            "namespace_correct": False,
            "doc_types": {},
            "namespaces": {},
            "unique_docs": 0,
        }
    
    scores = [chunk["score"] for chunk in chunks]
    namespaces = [chunk.get("namespace", "unknown") for chunk in chunks]
    detected_namespace = namespaces[0] if namespaces else "none"
    
    doc_types = defaultdict(int)
    for chunk in chunks:
        doc_type = chunk.get("metadata", {}).get("doc_type", "unknown")
        if isinstance(doc_type, str):
            doc_types[doc_type] += 1
    
    namespace_dist = defaultdict(int)
    for ns in namespaces:
        namespace_dist[ns] += 1
    
    unique_docs = len(set(chunk.get("doc_id", "") for chunk in chunks))
    
    namespace_correct = False
    if expected_namespace:
        namespace_correct = detected_namespace == expected_namespace
    
    return {
        "avg_score": np.mean(scores),
        "max_score": np.max(scores),
        "min_score": np.min(scores),
        "std_score": np.std(scores),
        "num_results": len(chunks),
        "detected_namespace": detected_namespace,
        "namespace_correct": namespace_correct,
        "expected_namespace": expected_namespace,
        "doc_types": dict(doc_types),
        "namespaces": dict(namespace_dist),
        "unique_docs": unique_docs,
    }


# ============================================================
# Strategy Comparison
# ============================================================

def compare_strategies(
    retriever: Retriever,
    queries: List[Dict[str, Any]],
    strategies: List[str],
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Compare all strategies across all queries.
    
    Returns:
        DataFrame with one row per query-strategy combination
    """
    
    results = []
    
    print(f"\n[EVALUATION] Testing {len(queries)} queries across {len(strategies)} strategies...")
    print("=" * 70)
    
    for query_info in tqdm(queries, desc="Queries"):
        query = query_info["query"]
        expected_namespace = query_info.get("expected_namespace")
        category = query_info.get("category", "general")
        
        for strategy in strategies:
            metrics = evaluate_retrieval(
                retriever=retriever,
                query=query,
                strategy=strategy,
                top_k=top_k,
                expected_namespace=expected_namespace,
            )
            
            results.append({
                "query": query,
                "category": category,
                "strategy": strategy,
                "expected_namespace": expected_namespace,
                "detected_namespace": metrics["detected_namespace"],
                "namespace_correct": metrics["namespace_correct"],
                "avg_score": metrics["avg_score"],
                "max_score": metrics["max_score"],
                "min_score": metrics["min_score"],
                "std_score": metrics["std_score"],
                "num_results": metrics["num_results"],
                "unique_docs": metrics["unique_docs"],
                "doc_types": json.dumps(metrics["doc_types"], ensure_ascii=False),
                "namespaces": json.dumps(metrics["namespaces"], ensure_ascii=False),
            })
    
    df = pd.DataFrame(results)
    return df


# ============================================================
# Statistics Computation
# ============================================================

def compute_strategy_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate statistics per strategy."""
    
    stats = df.groupby("strategy").agg({
        "avg_score": ["mean", "std", "min", "max"],
        "max_score": ["mean", "std"],
        "namespace_correct": "mean",
        "num_results": "mean",
        "unique_docs": "mean",
    }).round(4)
    
    stats.columns = ["_".join(col).strip() for col in stats.columns]
    stats = stats.reset_index()
    
    return stats


def compute_namespace_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute namespace detection accuracy."""
    
    namespace_df = df.groupby(["expected_namespace", "strategy"]).agg({
        "namespace_correct": ["mean", "count"],
    }).round(4)
    
    namespace_df.columns = ["accuracy", "count"]
    namespace_df = namespace_df.reset_index()
    
    return namespace_df


# ============================================================
# Main Evaluation Pipeline
# ============================================================

def run_evaluation(
    queries: List[Dict[str, Any]],
    output_dir: str,
    strategies: List[str],
    top_k: int = 5,
    api_keys_path: str = DEFAULT_API_KEYS_PATH,
):
    """
    Run evaluation and save results to CSV files.
    
    For visualization and analysis, use evaluate_chunking_strategies.ipynb
    which loads these CSV files.
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("CHUNKING STRATEGY EVALUATION")
    print("=" * 70)
    print(f"Queries: {len(queries)}")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Top-K: {top_k}")
    print("=" * 70 + "\n")
    
    # Initialize retriever
    print("[STEP] Initializing retriever...")
    retriever = Retriever(api_keys_path=api_keys_path)
    print("[OK] Retriever ready\n")
    
    # Run evaluation
    df = compare_strategies(retriever, queries, strategies, top_k)
    
    # Save raw results
    results_csv = output_path / "evaluation_results.csv"
    df.to_csv(results_csv, index=False, encoding="utf-8")
    print(f"\n[OK] Saved raw results to {results_csv}")
    
    # Compute statistics
    print("\n[STEP] Computing statistics...")
    strategy_stats = compute_strategy_statistics(df)
    namespace_stats = compute_namespace_accuracy(df)
    
    # Save statistics
    stats_csv = output_path / "strategy_statistics.csv"
    strategy_stats.to_csv(stats_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved strategy statistics to {stats_csv}")
    
    namespace_csv = output_path / "namespace_statistics.csv"
    namespace_stats.to_csv(namespace_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved namespace statistics to {namespace_csv}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_path}")
    print("\nKey Metrics:")
    print("-" * 70)
    for _, row in strategy_stats.iterrows():
        print(f"\n{row['strategy'].upper()}:")
        print(f"  Average Score: {row['avg_score_mean']:.4f} (±{row['avg_score_std']:.4f})")
        print(f"  Namespace Accuracy: {df[df['strategy']==row['strategy']]['namespace_correct'].mean():.2%}")
    print("\n" + "=" * 70)
    print("\n[INFO] For visualizations and detailed analysis, run evaluate_chunking_strategies.ipynb")
    print("       The notebook will load the CSV files from this directory.\n")


# ============================================================
# CLI
# ============================================================

def load_queries(file_path: str) -> List[Dict[str, Any]]:
    """Load queries from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "queries" in data:
            return data["queries"]
        else:
            raise ValueError(f"Invalid query file format: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate chunking strategies for RAG system (saves CSV results only)"
    )
    parser.add_argument(
        "--queries_file",
        type=str,
        default=None,
        help="JSON file with evaluation queries (optional, uses defaults if not provided)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["baseline", "sentence", "adaptive"],
        help="Chunking strategies to evaluate",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query",
    )
    parser.add_argument(
        "--api_keys_path",
        type=str,
        default=DEFAULT_API_KEYS_PATH,
        help="Path to API keys JSON file",
    )
    
    args = parser.parse_args()
    
    # Load queries
    if args.queries_file and os.path.exists(args.queries_file):
        queries = load_queries(args.queries_file)
        print(f"[INFO] Loaded {len(queries)} queries from {args.queries_file}")
    else:
        queries = DEFAULT_QUERIES
        print(f"[INFO] Using {len(queries)} default evaluation queries")
        if args.queries_file:
            print(f"[WARN] Query file not found: {args.queries_file}, using defaults")
    
    # Run evaluation
    run_evaluation(
        queries=queries,
        output_dir=args.output_dir,
        strategies=args.strategies,
        top_k=args.top_k,
        api_keys_path=args.api_keys_path,
    )


if __name__ == "__main__":
    main()
