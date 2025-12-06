"""
Chunking Strategy Evaluation Script
===================================

This script evaluates the performance of different chunking strategies:
- baseline
- sentence
- adaptive

It tests queries across all strategies and generates:
- Retrieval quality metrics
- Strategy comparison tables
- Visualizations
- Detailed evaluation report

Usage:
    python evaluate_chunking_strategies.py \
        --queries_file evaluation_queries.json \
        --output_dir ./evaluation_results \
        --top_k 5
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from retriever import Retriever, detect_namespace
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
        - chunks: The actual retrieved chunks
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
            "num_results": 0,
            "detected_namespace": "none",
            "namespace_correct": False,
            "doc_types": {},
            "namespaces": {},
            "unique_docs": 0,
            "chunks": [],
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
        "chunks": chunks,
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
# Analysis and Statistics
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
# Visualizations
# ============================================================

def plot_strategy_comparison(df: pd.DataFrame, output_dir: Path):
    """Create comparison plots for strategies."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10
    
    # 1. Average Score by Strategy
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Average score comparison
    ax = axes[0, 0]
    strategy_means = df.groupby("strategy")["avg_score"].mean().sort_values(ascending=False)
    bars = ax.bar(strategy_means.index, strategy_means.values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Average Retrieval Score", fontsize=11)
    ax.set_xlabel("Chunking Strategy", fontsize=11)
    ax.set_title("Average Retrieval Score by Strategy", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, (strategy, val) in enumerate(strategy_means.items()):
        ax.text(i, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    
    # Score distribution
    ax = axes[0, 1]
    for strategy in df["strategy"].unique():
        scores = df[df["strategy"] == strategy]["avg_score"]
        ax.hist(scores, alpha=0.6, label=strategy, bins=15)
    ax.set_xlabel("Average Retrieval Score", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Score Distribution by Strategy", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # Namespace accuracy
    ax = axes[1, 0]
    namespace_acc = df.groupby("strategy")["namespace_correct"].mean()
    bars = ax.bar(namespace_acc.index, namespace_acc.values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Namespace Detection Accuracy", fontsize=11)
    ax.set_xlabel("Chunking Strategy", fontsize=11)
    ax.set_title("Namespace Detection Accuracy by Strategy", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    for i, (strategy, val) in enumerate(namespace_acc.items()):
        ax.text(i, val + 0.02, f"{val:.2%}", ha="center", va="bottom", fontsize=9)
    
    # Unique documents per query
    ax = axes[1, 1]
    unique_docs_means = df.groupby("strategy")["unique_docs"].mean()
    bars = ax.bar(unique_docs_means.index, unique_docs_means.values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Average Unique Documents", fontsize=11)
    ax.set_xlabel("Chunking Strategy", fontsize=11)
    ax.set_title("Document Diversity by Strategy", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, (strategy, val) in enumerate(unique_docs_means.items()):
        ax.text(i, val + 0.1, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "strategy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved comparison plot to {output_path}")
    plt.close()


def plot_namespace_analysis(df: pd.DataFrame, output_dir: Path):
    """Create namespace analysis plots."""
    
    sns.set_style("whitegrid")
    
    # Namespace accuracy heatmap
    namespace_acc = compute_namespace_accuracy(df)
    pivot = namespace_acc.pivot(index="expected_namespace", columns="strategy", values="accuracy")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2%",
        cmap="YlOrRd",
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )
    ax.set_title("Namespace Detection Accuracy\n(Expected vs. Detected)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Chunking Strategy", fontsize=11)
    ax.set_ylabel("Expected Namespace", fontsize=11)
    
    plt.tight_layout()
    output_path = output_dir / "namespace_accuracy_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved namespace heatmap to {output_path}")
    plt.close()


def plot_category_analysis(df: pd.DataFrame, output_dir: Path):
    """Create category-based analysis plots."""
    
    sns.set_style("whitegrid")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    category_strategy = df.groupby(["category", "strategy"])["avg_score"].mean().reset_index()
    pivot = category_strategy.pivot(index="category", columns="strategy", values="avg_score")
    
    pivot.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.8)
    ax.set_ylabel("Average Retrieval Score", fontsize=11)
    ax.set_xlabel("Query Category", fontsize=11)
    ax.set_title("Retrieval Performance by Query Category and Strategy", fontsize=12, fontweight="bold")
    ax.legend(title="Strategy", title_fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    output_path = output_dir / "category_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved category analysis to {output_path}")
    plt.close()


# ============================================================
# Report Generation
# ============================================================

def generate_report(
    df: pd.DataFrame,
    strategy_stats: pd.DataFrame,
    namespace_stats: pd.DataFrame,
    output_dir: Path,
) -> str:
    """Generate markdown evaluation report."""
    
    report = []
    report.append("# Chunking Strategy Evaluation Report\n")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary
    report.append("## Summary\n")
    report.append(f"- **Total Queries**: {df['query'].nunique()}\n")
    report.append(f"- **Strategies Tested**: {', '.join(df['strategy'].unique())}\n")
    report.append(f"- **Total Tests**: {len(df)}\n\n")
    
    # Strategy Statistics
    report.append("## Strategy Comparison\n\n")
    report.append("### Overall Performance Metrics\n\n")
    report.append(strategy_stats.to_markdown(index=False))
    report.append("\n\n")
    
    # Best Strategy
    best_strategy = strategy_stats.loc[strategy_stats["avg_score_mean"].idxmax(), "strategy"]
    best_avg_score = strategy_stats["avg_score_mean"].max()
    report.append(f"**Best Performing Strategy**: `{best_strategy}` (avg score: {best_avg_score:.4f})\n\n")
    
    # Namespace Accuracy
    report.append("## Namespace Detection Accuracy\n\n")
    report.append("### Accuracy by Namespace and Strategy\n\n")
    report.append(namespace_stats.to_markdown(index=False))
    report.append("\n\n")
    
    overall_namespace_acc = df["namespace_correct"].mean()
    report.append(f"**Overall Namespace Detection Accuracy**: {overall_namespace_acc:.2%}\n\n")
    
    # Detailed Results
    report.append("## Detailed Results\n\n")
    report.append("### Sample Query Results\n\n")
    
    sample_queries = df.groupby("query").first().head(5)
    for _, row in sample_queries.iterrows():
        report.append(f"- **Query**: {row['query']}\n")
        report.append(f"  - Expected Namespace: `{row['expected_namespace']}`\n")
        report.append(f"  - Category: `{row['category']}`\n\n")
    
    # Recommendations
    report.append("## Recommendations\n\n")
    
    # Find best strategy per category
    category_best = df.groupby(["category", "strategy"])["avg_score"].mean().reset_index()
    category_best = category_best.loc[category_best.groupby("category")["avg_score"].idxmax()]
    
    report.append("### Best Strategy by Query Category\n\n")
    for _, row in category_best.iterrows():
        report.append(f"- **{row['category']}**: `{row['strategy']}` (avg score: {row['avg_score']:.4f})\n")
    report.append("\n")
    
    report.append("## Visualizations\n\n")
    report.append("- `strategy_comparison.png`: Overall strategy comparison\n")
    report.append("- `namespace_accuracy_heatmap.png`: Namespace detection accuracy\n")
    report.append("- `category_analysis.png`: Performance by query category\n\n")
    
    report_text = "\n".join(report)
    
    report_path = output_dir / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"[OK] Saved evaluation report to {report_path}")
    return report_text


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
    """Run complete evaluation pipeline."""
    
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
    
    # Generate visualizations
    print("\n[STEP] Generating visualizations...")
    plot_strategy_comparison(df, output_path)
    plot_namespace_analysis(df, output_path)
    plot_category_analysis(df, output_path)
    
    # Generate report
    print("\n[STEP] Generating evaluation report...")
    generate_report(df, strategy_stats, namespace_stats, output_path)
    
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
    print("\n" + "=" * 70 + "\n")


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
        description="Evaluate chunking strategies for RAG system"
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

