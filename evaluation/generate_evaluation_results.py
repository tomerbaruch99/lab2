"""
Chunking Strategy Evaluation Script
===================================

This script runs the evaluation of different chunking strategies:
- baseline
- sentence
- adaptive

It tests queries across all strategies and saves raw results to CSV files.
For visualization and analysis, use the analyze_evaluation_results.ipynb notebook or analyze_results.py script.

Usage:
    python generate_evaluation_results.py \
        --queries_file evaluation_queries.json \
        --output_dir evaluation/evaluation_results \
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

from retriever import Retriever, detect_namespace
from utils import (
    DEFAULT_API_KEYS_PATH,
    DEFAULT_TOP_K,
    DEFAULT_EVALUATION_STRATEGIES,
)

# Import baseline methods
try:
    from evaluation.baseline_methods import (
        TfIdfBaselineRetriever,
        RetrievalOnlyBaseline,
        KeywordMatchingBaseline,
    )
    BASELINES_AVAILABLE = True
except ImportError:
    BASELINES_AVAILABLE = False
    print("[WARN] Baseline methods not available. Install sklearn for TF-IDF baseline.")


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

def compute_retrieval_metrics(
    retrieved_chunks: List[Dict],
    ground_truth_texts: List[str],
    top_k: int
) -> Dict[str, float]:
    """
    Compute precision, recall, and accuracy metrics based on ground truth.
    
    Args:
        retrieved_chunks: List of retrieved chunks from retriever
        ground_truth_texts: List of ground truth relevant document texts
        top_k: Number of top chunks to consider
        
    Returns:
        Dictionary with precision, recall, and accuracy metrics
    """
    if not retrieved_chunks or not ground_truth_texts:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "relevant_retrieved": 0,
            "total_retrieved": 0,
            "total_relevant": len(ground_truth_texts),
        }
    
    # Take top_k chunks
    top_chunks = retrieved_chunks[:top_k]
    
    # Extract text from retrieved chunks
    retrieved_texts = []
    for chunk in top_chunks:
        text = chunk.get("chunk_text_only") or chunk.get("text", "")
        retrieved_texts.append(str(text).strip())
    
    # Check which retrieved texts match ground truth (simple text matching)
    # For more sophisticated matching, could use embeddings
    relevant_retrieved = 0
    for retrieved_text in retrieved_texts:
        for gt_text in ground_truth_texts:
            # Check if retrieved text contains significant portion of ground truth
            # or vice versa (simple overlap check)
            retrieved_lower = retrieved_text.lower()
            gt_lower = str(gt_text).lower()
            
            # Check for significant overlap (at least 30% of shorter text)
            min_len = min(len(retrieved_lower), len(gt_lower))
            if min_len == 0:
                continue
                
            # Simple word overlap check
            retrieved_words = set(retrieved_lower.split())
            gt_words = set(gt_lower.split())
            if len(retrieved_words) > 0 and len(gt_words) > 0:
                overlap = len(retrieved_words & gt_words)
                overlap_ratio = overlap / min(len(retrieved_words), len(gt_words))
                if overlap_ratio > 0.3:  # 30% word overlap threshold
                    relevant_retrieved += 1
                    break
    
    total_retrieved = len(top_chunks)
    total_relevant = len(ground_truth_texts)
    
    precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0
    recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
    accuracy = relevant_retrieved / max(total_retrieved, total_relevant) if max(total_retrieved, total_relevant) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "relevant_retrieved": relevant_retrieved,
        "total_retrieved": total_retrieved,
        "total_relevant": total_relevant,
    }


def evaluate_retrieval(
    retriever: Retriever,
    query: str,
    strategy: str,
    top_k: int = 5,
    expected_namespace: Optional[str] = None,
    ground_truth_documents: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Evaluate retrieval performance for a single query and chunking strategy.
    
    This function runs a retrieval query and collects comprehensive metrics about
    the results, including similarity scores, namespace detection accuracy,
    document diversity, and optional ground truth-based precision/recall.
    
    Args:
        retriever: Retriever instance configured with API keys and index
        query: User query string in Hebrew
        strategy: Chunking strategy to filter by ("baseline", "sentence", "adaptive")
        top_k: Number of top chunks to retrieve (default: 5)
        expected_namespace: Expected namespace for this query (for accuracy calculation)
        ground_truth_documents: Optional list of ground truth documents.
            Each dict should have 'text' and optionally 'label' fields.
            Used to calculate precision, recall, and accuracy metrics.
        
    Returns:
        Dictionary containing evaluation metrics:
        - avg_score: Average similarity score of retrieved chunks (0-1)
        - max_score: Highest similarity score among retrieved chunks
        - min_score: Lowest similarity score among retrieved chunks
        - std_score: Standard deviation of similarity scores
        - num_results: Number of chunks retrieved
        - detected_namespace: Namespace detected by retriever
        - namespace_correct: Boolean indicating if detected namespace matches expected
        - expected_namespace: The expected namespace (if provided)
        - doc_types: Dictionary mapping doc_type to count (e.g., {"pdf": 3, "html": 2})
        - namespaces: Dictionary mapping namespace to count in results
        - unique_docs: Number of unique documents represented in results
        - precision: Precision metric (if ground_truth_documents provided)
        - recall: Recall metric (if ground_truth_documents provided)
        - accuracy: Accuracy metric (if ground_truth_documents provided)
        
    Note:
        This function suppresses stdout during retrieval to avoid cluttering
        evaluation output. It captures all metrics from a single retrieval call.
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
    
    # If expected_namespace is not provided, infer it from the query
    if not expected_namespace:
        expected_namespace = detect_namespace(query)
    
    if not chunks:
        return {
            "avg_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "std_score": 0.0,
            "num_results": 0,
            "detected_namespace": "none",
            "namespace_correct": False,
            "expected_namespace": expected_namespace,
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
    
    namespace_correct = detected_namespace == expected_namespace
    
    result = {
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
    
    # Add ground truth metrics if available
    if ground_truth_documents:
        relevant_texts = [
            doc["text"] for doc in ground_truth_documents 
            if doc.get("label") == "relevant"
        ]
        if relevant_texts:
            metrics = compute_retrieval_metrics(chunks, relevant_texts, top_k)
            result.update(metrics)
    
    return result


# ============================================================
# Strategy Comparison
# ============================================================

def evaluate_baseline(
    baseline_retriever: Any,
    query: str,
    top_k: int = 5,
    expected_namespace: Optional[str] = None,
    baseline_type: str = "tfidf",
) -> Dict[str, Any]:
    """
    Evaluate a baseline retrieval method.
    
    Args:
        baseline_retriever: Baseline retriever instance
        query: Query text
        top_k: Number of results to retrieve
        expected_namespace: Expected namespace for accuracy calculation
        baseline_type: Type of baseline ("tfidf", "keyword", "retrieval_only")
    
    Returns:
        Dictionary with metrics (same format as evaluate_retrieval)
    """
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        if baseline_type == "retrieval_only":
            # For retrieval-only, we still need chunks
            if hasattr(baseline_retriever, 'retriever'):
                chunks = baseline_retriever.retriever.retrieve(
                    query=query,
                    top_k=top_k,
                    include_metadata=True,
                )
            else:
                chunks = []
        else:
            # For TF-IDF and keyword matching
            chunks = baseline_retriever.retrieve(
                query=query,
                top_k=top_k,
                include_metadata=True,
            )
    finally:
        sys.stdout = old_stdout
    
    # If expected_namespace is not provided, infer it from the query
    if not expected_namespace:
        expected_namespace = detect_namespace(query)
    
    if not chunks:
        return {
            "avg_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "std_score": 0.0,
            "num_results": 0,
            "detected_namespace": "none",
            "namespace_correct": False,
            "expected_namespace": expected_namespace,
            "doc_types": {},
            "namespaces": {},
            "unique_docs": 0,
        }
    
    scores = [chunk.get("score", 0.0) for chunk in chunks]
    namespaces = [chunk.get("namespace", "unknown") for chunk in chunks]
    detected_namespace = namespaces[0] if namespaces else "none"
    
    doc_types = defaultdict(int)
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if isinstance(metadata, dict):
            doc_type = metadata.get("doc_type", "unknown")
        else:
            doc_type = "unknown"
        if isinstance(doc_type, str):
            doc_types[doc_type] += 1
    
    namespace_dist = defaultdict(int)
    for ns in namespaces:
        namespace_dist[ns] += 1
    
    unique_docs = len(set(chunk.get("doc_id", "") for chunk in chunks))
    
    namespace_correct = detected_namespace == expected_namespace
    
    return {
        "avg_score": np.mean(scores) if scores else 0.0,
        "max_score": np.max(scores) if scores else 0.0,
        "min_score": np.min(scores) if scores else 0.0,
        "std_score": np.std(scores) if scores else 0.0,
        "num_results": len(chunks),
        "detected_namespace": detected_namespace,
        "namespace_correct": namespace_correct,
        "expected_namespace": expected_namespace,
        "doc_types": dict(doc_types),
        "namespaces": dict(namespace_dist),
        "unique_docs": unique_docs,
    }


def compare_strategies(
    retriever: Retriever,
    queries: List[Dict[str, Any]],
    strategies: List[str],
    top_k: int = 5,
    include_baselines: bool = False,
    api_keys_path: str = DEFAULT_API_KEYS_PATH,
) -> pd.DataFrame:
    """
    Compare all strategies across all queries, optionally including baselines.
    
    Args:
        retriever: Main RAG retriever
        queries: List of query dictionaries
        strategies: List of chunking strategies to test
        top_k: Number of results to retrieve
        include_baselines: If True, also evaluate baseline methods
        api_keys_path: Path to API keys file
        
    Returns:
        DataFrame with one row per query-strategy combination
    """
    
    results = []
    
    baseline_retrievers = {}
    if include_baselines and BASELINES_AVAILABLE:
        print("\n[INFO] Initializing baseline methods...")
        try:
            baseline_retrievers["tfidf"] = TfIdfBaselineRetriever(api_keys_path=api_keys_path)
            print("[OK] TF-IDF baseline ready")
        except Exception as e:
            print(f"[WARN] TF-IDF baseline unavailable: {e}")
        
        try:
            baseline_retrievers["keyword"] = KeywordMatchingBaseline()
            print("[OK] Keyword matching baseline ready")
        except Exception as e:
            print(f"[WARN] Keyword baseline unavailable: {e}")
        
        try:
            baseline_retrievers["retrieval_only"] = RetrievalOnlyBaseline(api_keys_path=api_keys_path)
            print("[OK] Retrieval-only baseline ready")
        except Exception as e:
            print(f"[WARN] Retrieval-only baseline unavailable: {e}")
    
    print(f"\n[EVALUATION] Testing {len(queries)} queries across {len(strategies)} strategies...")
    if include_baselines:
        print(f"          Plus {len(baseline_retrievers)} baseline method(s)")
    print("=" * 70)
    
    for query_info in tqdm(queries, desc="Queries"):
        query = query_info.get("query") or query_info.get("question", "")
        expected_namespace = query_info.get("expected_namespace")
        category = query_info.get("category", "general")
        ground_truth_documents = query_info.get("ground_truth_documents")
        
        # Test main strategies
        for strategy in strategies:
            metrics = evaluate_retrieval(
                retriever=retriever,
                query=query,
                strategy=strategy,
                top_k=top_k,
                expected_namespace=expected_namespace,
                ground_truth_documents=ground_truth_documents,
            )
            
            result_row = {
                "query": query,
                "category": category,
                "strategy": strategy,
                "expected_namespace": metrics["expected_namespace"],  # Use inferred value if original was None
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
            }
            
            # Add ground truth metrics if available
            if "precision" in metrics:
                result_row["precision"] = metrics["precision"]
                result_row["recall"] = metrics["recall"]
                result_row["accuracy"] = metrics["accuracy"]
                result_row["relevant_retrieved"] = metrics["relevant_retrieved"]
                result_row["total_relevant"] = metrics["total_relevant"]
            
            results.append(result_row)
        
        # Test baseline methods
        if include_baselines:
            for baseline_name, baseline_retriever in baseline_retrievers.items():
                metrics = evaluate_baseline(
                    baseline_retriever=baseline_retriever,
                    query=query,
                    top_k=top_k,
                    expected_namespace=expected_namespace,
                    baseline_type=baseline_name,
                )
                
                results.append({
                    "query": query,
                    "category": category,
                    "strategy": f"baseline_{baseline_name}",
                    "expected_namespace": metrics["expected_namespace"],  # Use inferred value if original was None
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
    
    agg_dict = {
        "avg_score": ["mean", "std", "min", "max"],
        "max_score": ["mean", "std"],
        "namespace_correct": "mean",
        "num_results": "mean",
        "unique_docs": "mean",
    }
    
    # Add ground truth metrics if available
    if "precision" in df.columns:
        agg_dict["precision"] = ["mean", "std"]
        agg_dict["recall"] = ["mean", "std"]
        agg_dict["accuracy"] = ["mean", "std"]
    
    stats = df.groupby("strategy").agg(agg_dict).round(4)
    
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


def load_testset_with_ground_truth(file_path: str) -> List[Dict[str, Any]]:
    """
    Load testset format with ground truth documents.
    
    Expected format:
    {
        "queries": [
            {
                "question": "query text",
                "documents": [
                    {"text": "...", "label": "relevant"},
                    {"text": "...", "label": "irrelevant"}
                ]
            }
        ]
    }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    queries = []
    if isinstance(data, dict) and "queries" in data:
        query_list = data["queries"]
    elif isinstance(data, list):
        query_list = data
    else:
        raise ValueError(f"Invalid testset format: {file_path}")
    
    for item in query_list:
        query_text = item.get("question") or item.get("query", "")
        documents = item.get("documents", [])
        
        queries.append({
            "query": query_text,
            "question": query_text,
            "ground_truth_documents": documents,
            "category": item.get("category", "general"),
            "expected_namespace": item.get("expected_namespace"),
        })
    
    return queries


# ============================================================
# Quantitative Analysis Functions
# ============================================================

def compute_score_distribution(df: pd.DataFrame, score_col: str = "avg_score") -> pd.DataFrame:
    """
    Compute score distribution with explicit proportions in each rank category.
    
    Categories:
    - Excellent: >= 0.8
    - Good: 0.6-0.8
    - Moderate: 0.4-0.6
    - Poor: < 0.4
    """
    def categorize_score(score):
        if score >= 0.8:
            return "Excellent (≥0.8)"
        elif score >= 0.6:
            return "Good (0.6-0.8)"
        elif score >= 0.4:
            return "Moderate (0.4-0.6)"
        else:
            return "Poor (<0.4)"
    
    df_copy = df.copy()
    df_copy["score_category"] = df_copy[score_col].apply(categorize_score)
    
    distribution = df_copy.groupby(["strategy", "score_category"]).size().reset_index(name="count")
    total_per_strategy = df_copy.groupby("strategy").size().reset_index(name="total")
    
    distribution = distribution.merge(total_per_strategy, on="strategy")
    distribution["proportion"] = (distribution["count"] / distribution["total"] * 100).round(2)
    
    return distribution


def compute_improvement_over_baseline(
    df: pd.DataFrame,
    main_strategies: List[str],
    baseline_strategies: List[str],
    metric: str = "avg_score",
) -> pd.DataFrame:
    """
    Compute improvement percentages over baseline methods.
    
    Returns DataFrame with improvement metrics for each main strategy vs each baseline.
    """
    improvements = []
    
    main_df = df[df["strategy"].isin(main_strategies)]
    baseline_df = df[df["strategy"].isin(baseline_strategies)]
    
    for main_strat in main_strategies:
        main_scores = main_df[main_df["strategy"] == main_strat][metric].values
        
        for baseline_strat in baseline_strategies:
            baseline_scores = baseline_df[baseline_df["strategy"] == baseline_strat][metric].values
            
            # Align by query (assuming same queries tested for both)
            if len(main_scores) == len(baseline_scores):
                improvements_raw = ((main_scores - baseline_scores) / baseline_scores * 100)
                improvements_raw = np.where(baseline_scores == 0, np.nan, improvements_raw)
                
                mean_improvement = np.nanmean(improvements_raw)
                median_improvement = np.nanmedian(improvements_raw)
                std_improvement = np.nanstd(improvements_raw)
                num_better = np.sum(main_scores > baseline_scores)
                num_worse = np.sum(main_scores < baseline_scores)
                num_equal = np.sum(main_scores == baseline_scores)
                total = len(main_scores)
                
                improvements.append({
                    "main_strategy": main_strat,
                    "baseline_strategy": baseline_strat,
                    "mean_improvement_pct": round(mean_improvement, 2),
                    "median_improvement_pct": round(median_improvement, 2),
                    "std_improvement_pct": round(std_improvement, 2),
                    "queries_better": num_better,
                    "queries_worse": num_worse,
                    "queries_equal": num_equal,
                    "total_queries": total,
                    "proportion_better_pct": round(num_better / total * 100, 2) if total > 0 else 0,
                })
    
    return pd.DataFrame(improvements)


def statistical_significance_test(
    df: pd.DataFrame,
    strategy1: str,
    strategy2: str,
    metric: str = "avg_score",
) -> Dict[str, Any]:
    """
    Perform paired t-test to compare two strategies.
    
    Returns:
        Dictionary with test results including p-value, t-statistic, and interpretation
    """
    try:
        from scipy import stats
    except ImportError:
        return {
            "error": "scipy not available for statistical tests",
            "strategy1": strategy1,
            "strategy2": strategy2,
        }
    
    # Get scores for each query
    df1 = df[df["strategy"] == strategy1].sort_values("query")
    df2 = df[df["strategy"] == strategy2].sort_values("query")
    
    scores1 = df1[metric].values
    scores2 = df2[metric].values
    
    if len(scores1) != len(scores2):
        return {
            "error": f"Mismatched query counts: {len(scores1)} vs {len(scores2)}",
            "strategy1": strategy1,
            "strategy2": strategy2,
        }
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    mean_diff = np.mean(scores1 - scores2)
    mean1 = np.mean(scores1)
    mean2 = np.mean(scores2)
    
    # Interpretation
    alpha = 0.05
    significant = p_value < alpha
    interpretation = (
        f"{'Significant' if significant else 'Not significant'} difference "
        f"(p={p_value:.4f}, α={alpha})"
    )
    
    return {
        "strategy1": strategy1,
        "strategy2": strategy2,
        "metric": metric,
        "mean1": round(mean1, 4),
        "mean2": round(mean2, 4),
        "mean_difference": round(mean_diff, 4),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant": significant,
        "interpretation": interpretation,
        "n_samples": len(scores1),
    }


def compute_comparative_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive comparative statistics including:
    - Best performing strategy overall
    - Performance vs baselines
    - Statistical significance tests
    - Score distributions
    """
    # Identify main strategies vs baselines
    all_strategies = df["strategy"].unique().tolist()
    main_strategies = [s for s in all_strategies if not s.startswith("baseline_")]
    baseline_strategies = [s for s in all_strategies if s.startswith("baseline_")]
    
    results = {
        "main_strategies": main_strategies,
        "baseline_strategies": baseline_strategies,
    }
    
    # Overall statistics
    strategy_stats = compute_strategy_statistics(df)
    results["overall_statistics"] = strategy_stats.to_dict("records")
    
    # Best strategy by average score
    if len(strategy_stats) > 0:
        best_strategy = strategy_stats.loc[strategy_stats["avg_score_mean"].idxmax()]
        results["best_strategy"] = {
            "strategy": best_strategy["strategy"],
            "avg_score": best_strategy["avg_score_mean"],
            "std_score": best_strategy.get("avg_score_std", 0),
        }
    
    # Improvement over baselines
    if baseline_strategies and main_strategies:
        improvements = compute_improvement_over_baseline(
            df, main_strategies, baseline_strategies
        )
        results["improvements_over_baselines"] = improvements.to_dict("records")
        
        # Best improvement for each main strategy
        best_improvements = []
        for main_strat in main_strategies:
            main_impr = improvements[improvements["main_strategy"] == main_strat]
            if len(main_impr) > 0:
                best_impr = main_impr.loc[main_impr["mean_improvement_pct"].idxmax()]
                best_improvements.append({
                    "strategy": main_strat,
                    "vs_baseline": best_impr["baseline_strategy"],
                    "improvement_pct": best_impr["mean_improvement_pct"],
                    "proportion_better_pct": best_impr["proportion_better_pct"],
                })
        results["best_improvements"] = best_improvements
    
    # Score distribution
    score_dist = compute_score_distribution(df)
    results["score_distribution"] = score_dist.to_dict("records")
    
    # Statistical significance tests (main strategies vs best baseline)
    if baseline_strategies and main_strategies:
        # Find best baseline
        baseline_stats = strategy_stats[strategy_stats["strategy"].isin(baseline_strategies)]
        if len(baseline_stats) > 0:
            best_baseline = baseline_stats.loc[baseline_stats["avg_score_mean"].idxmax()]
            best_baseline_name = best_baseline["strategy"]
            
            significance_tests = []
            for main_strat in main_strategies:
                test_result = statistical_significance_test(
                    df, main_strat, best_baseline_name
                )
                significance_tests.append(test_result)
            results["significance_tests_vs_best_baseline"] = significance_tests
    
    return results


# ============================================================
# Main Evaluation Pipeline
# ============================================================

def run_evaluation(
    queries: List[Dict[str, Any]],
    output_dir: str,
    strategies: List[str],
    top_k: int = 5,
    api_keys_path: str = DEFAULT_API_KEYS_PATH,
    include_baselines: bool = False,
    index_name: Optional[str] = None,
):
    """
    Run evaluation and save results to CSV files.
    
    This function:
    - Queries Pinecone and Gemini APIs (requires API keys)
    - Saves all results to CSV files locally
    - Does NOT generate visualizations or analysis
    
    For visualization and analysis (NO API access required), use:
    - analyze_evaluation_results.ipynb (Jupyter notebook)
    - analyze_results.py (command-line script)
    Both analyze locally stored CSV files only.
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
    if index_name:
        print(f"[INFO] Using index: {index_name}")
    retriever = Retriever(api_keys_path=api_keys_path, index_name=index_name)
    print("[OK] Retriever ready\n")
    
    # Run evaluation
    df = compare_strategies(
        retriever, 
        queries, 
        strategies, 
        top_k,
        include_baselines=include_baselines,
        api_keys_path=api_keys_path,
    )
    
    # Save raw results
    results_csv = output_path / "evaluation_results.csv"
    df.to_csv(results_csv, index=False, encoding="utf-8")
    print(f"\n[OK] Saved raw results to {results_csv}")
    
    # Compute statistics
    print("\n[STEP] Computing statistics...")
    strategy_stats = compute_strategy_statistics(df)
    namespace_stats = compute_namespace_accuracy(df)
    
    # Compute comprehensive quantitative analysis
    print("[STEP] Computing quantitative comparative analysis...")
    comparative_stats = compute_comparative_statistics(df)
    score_distribution = compute_score_distribution(df)
    
    # Save statistics
    stats_csv = output_path / "strategy_statistics.csv"
    strategy_stats.to_csv(stats_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved strategy statistics to {stats_csv}")
    
    namespace_csv = output_path / "namespace_statistics.csv"
    namespace_stats.to_csv(namespace_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved namespace statistics to {namespace_csv}")
    
    # Save quantitative analysis
    score_dist_csv = output_path / "score_distribution.csv"
    score_distribution.to_csv(score_dist_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved score distribution to {score_dist_csv}")
    
    # Save improvements over baselines if baselines were included
    if comparative_stats.get("improvements_over_baselines"):
        improvements_df = pd.DataFrame(comparative_stats["improvements_over_baselines"])
        improvements_csv = output_path / "improvements_over_baselines.csv"
        improvements_df.to_csv(improvements_csv, index=False, encoding="utf-8")
        print(f"[OK] Saved baseline improvements to {improvements_csv}")
    
    # Save significance tests if available
    if comparative_stats.get("significance_tests_vs_best_baseline"):
        sig_tests_df = pd.DataFrame(comparative_stats["significance_tests_vs_best_baseline"])
        sig_tests_csv = output_path / "statistical_significance_tests.csv"
        sig_tests_df.to_csv(sig_tests_csv, index=False, encoding="utf-8")
        print(f"[OK] Saved statistical significance tests to {sig_tests_csv}")
    
    # Generate quantitative report
    try:
        # Try relative import first
        try:
            from .quantitative_report_generator import generate_quantitative_report
        except ImportError:
            # Fallback to absolute import
            from evaluation.quantitative_report_generator import generate_quantitative_report
        
        quantitative_report_path = output_path / "quantitative_analysis_report.md"
        generate_quantitative_report(
            comparative_stats, 
            strategy_stats, 
            score_distribution,
            output_path=quantitative_report_path,
        )
        print(f"[OK] Saved quantitative analysis report to {quantitative_report_path}")
    except ImportError as e:
        # Fallback if module not found (shouldn't happen but handle gracefully)
        print(f"[WARN] Could not generate quantitative report: {e}")
    
    # Print summary with quantitative context
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
        if "precision_mean" in row:
            print(f"  Precision: {row['precision_mean']:.4f} (±{row.get('precision_std', 0):.4f})")
            print(f"  Recall: {row['recall_mean']:.4f} (±{row.get('recall_std', 0):.4f})")
            print(f"  Accuracy: {row['accuracy_mean']:.4f} (±{row.get('accuracy_std', 0):.4f})")
    
    # Print quantitative comparisons
    if comparative_stats.get("best_strategy"):
        best = comparative_stats["best_strategy"]
        print(f"\n{'=' * 70}")
        print("QUANTITATIVE ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"\nBest Performing Strategy: {best['strategy'].upper()}")
        print(f"  Average Score: {best['avg_score']:.4f} (±{best['std_score']:.4f})")
        
        if comparative_stats.get("best_improvements"):
            print(f"\nImprovements Over Baselines:")
            for impr in comparative_stats["best_improvements"]:
                print(f"  {impr['strategy'].upper()} vs {impr['vs_baseline']}:")
                print(f"    +{impr['improvement_pct']:.2f}% average improvement")
                print(f"    {impr['proportion_better_pct']:.1f}% of queries performed better")
        
        if comparative_stats.get("score_distribution"):
            print(f"\nScore Distribution (Proportion of Queries):")
            for strat in strategy_stats["strategy"].unique():
                strat_dist = score_distribution[score_distribution["strategy"] == strat]
                if len(strat_dist) > 0:
                    print(f"\n  {strat.upper()}:")
                    for _, dist_row in strat_dist.iterrows():
                        print(f"    {dist_row['score_category']}: {dist_row['proportion']:.1f}% ({dist_row['count']}/{dist_row['total']} queries)")
        
        if comparative_stats.get("significance_tests_vs_best_baseline"):
            print(f"\nStatistical Significance Tests (vs Best Baseline):")
            for test in comparative_stats["significance_tests_vs_best_baseline"]:
                if "error" not in test:
                    print(f"  {test['strategy1']} vs {test['strategy2']}:")
                    print(f"    Mean difference: {test['mean_difference']:+.4f}")
                    print(f"    {test['interpretation']}")
    
    print("\n" + "=" * 70)
    print("\n[INFO] For visualizations and detailed analysis, run analyze_evaluation_results.ipynb or analyze_results.py")
    print("       The notebook will load the CSV files from this directory.")
    print(f"       See quantitative_analysis_report.md for comprehensive quantitative analysis.\n")


# ============================================================
# CLI
# ============================================================

def load_queries(file_path: str) -> List[Dict[str, Any]]:
    """
    Load queries from JSON file.
    Supports both evaluation_queries.json format and embedding_testset.json format.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Check if it's the testset format (with ground truth documents)
    if isinstance(data, dict) and "queries" in data:
        # Check if it has documents with labels (testset format)
        if data["queries"] and "documents" in data["queries"][0]:
            return load_testset_with_ground_truth(file_path)
        else:
            # Regular queries format
            return data["queries"]
    elif isinstance(data, list):
        # Check if first item has "documents" field (testset format)
        if data and isinstance(data[0], dict) and "documents" in data[0]:
            return load_testset_with_ground_truth(file_path)
        else:
            # Regular queries format
            return data
    else:
        raise ValueError(f"Invalid query file format: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation results by querying APIs (saves CSV results for later analysis)"
    )
    parser.add_argument(
        "--queries_file",
        type=str,
        default=None,
        help="JSON file with evaluation queries (optional, uses defaults if not provided). "
             "Supports both evaluation_queries.json and tests/embedding_testset.json formats.",
    )
    parser.add_argument(
        "--testset_file",
        type=str,
        default=None,
        help="Path to testset with ground truth labels (tests/embedding_testset.json). "
             "If provided, computes precision/recall metrics.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/evaluation_results",
        help="Output directory for evaluation results (default: evaluation/evaluation_results)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=DEFAULT_EVALUATION_STRATEGIES,
        help="Chunking strategies to evaluate",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve per query",
    )
    parser.add_argument(
        "--api_keys_path",
        type=str,
        default=DEFAULT_API_KEYS_PATH,
        help="Path to API keys JSON file",
    )
    parser.add_argument(
        "--include_baselines",
        action="store_true",
        help="Include baseline methods (TF-IDF, keyword matching, retrieval-only) in evaluation",
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default=None,
        help="Pinecone index name to use (default: from utils config or 'haifa-rag')",
    )
    
    args = parser.parse_args()
    
    # Load queries
    if args.testset_file and os.path.exists(args.testset_file):
        queries = load_testset_with_ground_truth(args.testset_file)
        print(f"[INFO] Loaded {len(queries)} queries with ground truth from {args.testset_file}")
        print(f"[INFO] Will compute precision/recall metrics based on ground truth labels")
    elif args.queries_file and os.path.exists(args.queries_file):
        queries = load_queries(args.queries_file)
        print(f"[INFO] Loaded {len(queries)} queries from {args.queries_file}")
        # Check if queries have ground truth
        if queries and queries[0].get("ground_truth_documents"):
            print(f"[INFO] Ground truth documents detected - will compute precision/recall metrics")
    else:
        queries = DEFAULT_QUERIES
        print(f"[INFO] Using {len(queries)} default evaluation queries")
        if args.queries_file or args.testset_file:
            print(f"[WARN] Query/testset file not found, using defaults")
    
    # Run evaluation
    run_evaluation(
        queries=queries,
        output_dir=args.output_dir,
        strategies=args.strategies,
        top_k=args.top_k,
        api_keys_path=args.api_keys_path,
        include_baselines=args.include_baselines,
        index_name=args.index_name,
    )


if __name__ == "__main__":
    main()
