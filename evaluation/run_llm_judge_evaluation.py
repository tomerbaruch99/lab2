"""
LLM Judge Evaluation Script
============================

This script evaluates RAG answers using LLM-as-a-judge by:
1. Loading queries from embedding_testset.json
2. Generating RAG answers for each query (using DEFAULT_GEMINI_MODEL from config)
3. Creating gold answers from relevant documents in the testset
4. Using llm_judge.py to evaluate answer quality (using DEFAULT_LLM_JUDGE_MODEL from config)
5. Saving results to CSV

Note: The script uses different Gemini models for RAG generation and judging
to ensure unbiased evaluation. The judge model is independent from the model
being evaluated. Models are configured in utils/config.py.

Usage:
    # Standard evaluation
    python evaluation/run_llm_judge_evaluation.py \
        --testset_file tests/embedding_testset.json \
        --output_dir evaluation/llm_judge_results \
        --strategies baseline sentence adaptive \
        --top_k 5

    # Test all 4 combinations of enrichment/reranking
    # IMPORTANT: First determine your best strategy/chunk_size/K from previous evaluations
    # Then use that configuration here
    python evaluation/run_llm_judge_evaluation.py \
        --testset_file tests/embedding_testset.json \
        --output_dir evaluation/llm_judge_results \
        --strategies adaptive \
        --top_k 5 \
        --test_enrichment_reranking \
        --baseline_strategy adaptive  # Use your best performing strategy here
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any

import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gemini_integration import GeminiRAG, init_gemini, load_api_keys
from evaluation.llm_judge import judge_answer
from utils import DEFAULT_API_KEYS_PATH, DEFAULT_TOP_K, DEFAULT_EVALUATION_STRATEGIES, DEFAULT_LLM_JUDGE_MODEL


def load_testset(file_path: str) -> List[Dict[str, Any]]:
    """Load queries from embedding_testset.json format."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "queries" in data:
        return data["queries"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Invalid testset format: {file_path}")


def create_gold_answer_from_documents(documents: List[Dict]) -> str:
    """
    Create a gold answer by combining all relevant documents.
    
    Args:
        documents: List of documents with 'text' and 'label' fields
        
    Returns:
        Combined text from all relevant documents
    """
    relevant_texts = [
        doc["text"] for doc in documents 
        if doc.get("label") == "relevant"
    ]
    
    if not relevant_texts:
        return "לא נמצאו מסמכים רלוונטיים."
    
    # Combine relevant documents with newlines
    return "\n\n".join(relevant_texts)


def evaluate_with_llm_judge(
    rag: GeminiRAG,
    gemini_model,
    query: str,
    gold_answer: str,
    strategy: str,
    top_k: int = 5,
    use_query_enhancement: bool = False,
    use_reranking: bool = False,
) -> Dict[str, Any]:
    """
    Generate RAG answer and evaluate it using LLM judge.
    
    Args:
        rag: GeminiRAG instance
        gemini_model: Initialized Gemini model for judging
        query: User question
        gold_answer: Gold standard answer
        strategy: Chunking strategy to use
        top_k: Number of chunks to retrieve
        use_query_enhancement: Whether to use query enrichment
        use_reranking: Whether to use reranking
        
    Returns:
        Dictionary with RAG answer and judge scores
    """
    # Generate RAG answer
    try:
        # Get chunks for debugging
        result_with_chunks = rag.answer_question(
            question=query,
            top_k=top_k,
            strategy=strategy,
            return_chunks=True,
            use_query_enhancement=use_query_enhancement,
            use_reranking=use_reranking,
        )
        rag_answer = result_with_chunks.get("answer", "")
        chunks = result_with_chunks.get("chunks", [])
        
        # Debug: Check retrieval
        config_label = f"{strategy}"
        if use_query_enhancement:
            config_label += "+enrich"
        if use_reranking:
            config_label += "+rerank"
        
        if not chunks:
            print(f"[WARN] No chunks retrieved for query: {query[:50]}... (config: {config_label})")
        elif len(chunks) < top_k:
            print(f"[WARN] Only {len(chunks)}/{top_k} chunks retrieved for query: {query[:50]}... (config: {config_label})")
        
        # Debug: Show similarity scores
        if chunks:
            scores_list = [chunk.get('score', 0.0) for chunk in chunks]
            avg_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
            print(f"[DEBUG] Retrieved {len(chunks)} chunks (avg similarity: {avg_score:.3f}, max: {max(scores_list):.3f}, min: {min(scores_list):.3f})")
        
        # Debug: Check if RAG answer is empty or indicates no information
        if not rag_answer or "איני יכול" in rag_answer or "לא נמצא" in rag_answer or "מצטער" in rag_answer:
            print(f"[WARN] RAG answer indicates no information found for query: {query[:50]}...")
            print(f"[WARN] RAG answer: {rag_answer[:150]}...")
            if chunks:
                print(f"[WARN] But {len(chunks)} chunks were retrieved. Top chunk preview: {chunks[0].get('chunk_text_only', chunks[0].get('text', ''))[:100]}...")
                print(f"[WARN] Top chunk similarity score: {chunks[0].get('score', 0.0):.3f}")
                # Show all chunk scores
                chunk_scores = [f"{c.get('score', 0.0):.3f}" for c in chunks]
                print(f"[WARN] All chunk scores: {chunk_scores}")
    except Exception as e:
        print(f"[WARN] Error generating RAG answer: {e}")
        import traceback
        traceback.print_exc()
        rag_answer = ""
    
    # Judge the answer (even if RAG answer is empty or indicates no info)
    if rag_answer:
        scores = judge_answer(
            question=query,
            gold_answer=gold_answer,
            rag_answer=rag_answer,
            gemini_model=gemini_model,
        )
    else:
        # If no RAG answer, still try to judge (with empty answer)
        scores = judge_answer(
            question=query,
            gold_answer=gold_answer,
            rag_answer="לא נמצא מידע רלוונטי.",
            gemini_model=gemini_model,
        )
    
    return {
        "rag_answer": rag_answer,
        **scores,
    }


def run_llm_judge_evaluation(
    testset_file: str,
    output_dir: str,
    strategies: List[str],
    top_k: int = 5,
    api_keys_path: str = DEFAULT_API_KEYS_PATH,
    index_name: Optional[str] = None,
    gemini_model_name: str = DEFAULT_LLM_JUDGE_MODEL,
    test_enrichment_reranking: bool = False,
    baseline_strategy: Optional[str] = None,
):
    """
    Run LLM judge evaluation on testset.
    
    Args:
        testset_file: Path to embedding_testset.json
        output_dir: Directory to save results
        strategies: List of chunking strategies to test
        top_k: Number of chunks to retrieve
        api_keys_path: Path to API keys file
        index_name: Pinecone index name (optional)
        gemini_model_name: Gemini model for judging
        test_enrichment_reranking: If True, test all 4 combinations of enrichment/reranking
        baseline_strategy: Strategy to use as baseline for comparison. 
                          If None and test_enrichment_reranking=True, uses first strategy in list.
                          Note: You should determine the best strategy/chunk_size/K from previous 
                          evaluations before running enrichment/reranking tests.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("LLM-AS-A-JUDGE EVALUATION")
    print("=" * 70)
    print(f"Testset: {testset_file}")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Top-K: {top_k}")
    if test_enrichment_reranking:
        # Determine baseline strategy
        if baseline_strategy is None:
            baseline_strategy = strategies[0] if strategies else "adaptive"
            print(f"[NOTE] No baseline_strategy specified, using first strategy: {baseline_strategy}")
        print(f"Testing enrichment/reranking: YES (4 combinations per strategy)")
        print(f"Baseline for comparison: {baseline_strategy} (no enrichment, no reranking)")
        print(f"[NOTE] Make sure {baseline_strategy} is your best performing strategy/chunk_size/K")
        print(f"[NOTE] from previous evaluations before running this test.")
    else:
        print(f"Testing enrichment/reranking: NO")
    print("=" * 70 + "\n")
    
    # Load testset
    print("[STEP] Loading testset...")
    queries = load_testset(testset_file)
    print(f"[OK] Loaded {len(queries)} queries\n")
    
    # Initialize RAG system (uses DEFAULT_GEMINI_MODEL from config)
    print("[STEP] Initializing RAG system...")
    from utils import DEFAULT_GEMINI_MODEL
    rag = GeminiRAG(
        api_keys_path=api_keys_path,
        index_name=index_name,
    )
    print(f"[OK] RAG system ready (using {DEFAULT_GEMINI_MODEL} for answer generation)\n")
    
    # Initialize Gemini model for judging (uses different model)
    print("[STEP] Initializing Gemini model for judging...")
    api_keys = load_api_keys(api_keys_path)
    gemini_model = init_gemini(api_keys, gemini_model_name)
    print(f"[OK] Gemini model '{gemini_model_name}' ready for judging\n")
    
    # Run evaluation
    if test_enrichment_reranking:
        num_configs = len(strategies) * 4  # 4 combinations per strategy
        print(f"[EVALUATION] Evaluating {len(queries)} queries across {len(strategies)} strategies × 4 combinations = {num_configs} total configs...")
    else:
        print(f"[EVALUATION] Evaluating {len(queries)} queries across {len(strategies)} strategies...")
    print("=" * 70)
    
    results = []
    
    # Define all combinations to test
    if test_enrichment_reranking:
        combinations = [
            (False, False, "baseline"),
            (True, False, "enrichment_only"),
            (False, True, "reranking_only"),
            (True, True, "both"),
        ]
    else:
        combinations = [(False, False, "baseline")]
    
    for query_info in tqdm(queries, desc="Queries"):
        query = query_info.get("query") or query_info.get("question", "")
        documents = query_info.get("documents", [])
        
        # Create gold answer from relevant documents
        gold_answer = create_gold_answer_from_documents(documents)
        
        # Evaluate with each strategy and combination
        for strategy in strategies:
            for use_enrich, use_rerank, combo_label in combinations:
                eval_result = evaluate_with_llm_judge(
                    rag=rag,
                    gemini_model=gemini_model,
                    query=query,
                    gold_answer=gold_answer,
                    strategy=strategy,
                    top_k=top_k,
                    use_query_enhancement=use_enrich,
                    use_reranking=use_rerank,
                )
                
                # Create configuration identifier
                config_name = strategy
                if test_enrichment_reranking:
                    config_name = f"{strategy}_{combo_label}"
                
                results.append({
                    "query": query,
                    "strategy": strategy,
                    "config": config_name,
                    "use_query_enhancement": use_enrich,
                    "use_reranking": use_rerank,
                    "combination": combo_label,
                    "gold_answer": gold_answer,
                    "rag_answer": eval_result["rag_answer"],
                    "correctness": eval_result["correctness"],
                    "faithfulness": eval_result["faithfulness"],
                    "completeness": eval_result["completeness"],
                    "conciseness": eval_result["conciseness"],
                    "overall": eval_result["overall"],
                })
    
    # Save results
    df = pd.DataFrame(results)
    results_csv = output_path / "llm_judge_results.csv"
    df.to_csv(results_csv, index=False, encoding="utf-8")
    print(f"\n[OK] Saved results to {results_csv}")
    
    # Compute statistics
    print("\n[STEP] Computing statistics...")
    
    # Group by config (which includes combination label if testing enrichment/reranking)
    group_by_col = "config" if test_enrichment_reranking else "strategy"
    stats = df.groupby(group_by_col).agg({
        "correctness": ["mean", "std"],
        "faithfulness": ["mean", "std"],
        "completeness": ["mean", "std"],
        "conciseness": ["mean", "std"],
        "overall": ["mean", "std"],
    }).round(4)
    
    stats.columns = ["_".join(col).strip() for col in stats.columns]
    stats = stats.reset_index()
    
    stats_csv = output_path / "llm_judge_statistics.csv"
    stats.to_csv(stats_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved statistics to {stats_csv}")
    
    # If testing enrichment/reranking, create comparison against baseline
    if test_enrichment_reranking:
        print("\n[STEP] Computing comparison against baseline...")
        # Determine baseline strategy if not set
        if baseline_strategy is None:
            baseline_strategy = strategies[0] if strategies else "adaptive"
        baseline_config = f"{baseline_strategy}_baseline"
        
        # Get baseline scores
        baseline_df = df[df["config"] == baseline_config].copy()
        if len(baseline_df) > 0:
            comparison_results = []
            
            for config in df[group_by_col].unique():
                if config == baseline_config:
                    continue
                
                config_df = df[df[group_by_col] == config].copy()
                
                # Merge on query to compare same queries
                merged = baseline_df.merge(
                    config_df,
                    on="query",
                    suffixes=("_baseline", "_config")
                )
                
                if len(merged) > 0:
                    # Calculate improvements
                    metrics = ["correctness", "faithfulness", "completeness", "conciseness", "overall"]
                    for metric in metrics:
                        baseline_col = f"{metric}_baseline"
                        config_col = f"{metric}_config"
                        
                        improvements = merged[config_col] - merged[baseline_col]
                        mean_improvement = improvements.mean()
                        median_improvement = improvements.median()
                        better_count = (improvements > 0).sum()
                        total_count = len(improvements)
                        
                        comparison_results.append({
                            "config": config,
                            "baseline_config": baseline_config,
                            "metric": metric,
                            "mean_improvement": round(mean_improvement, 4),
                            "median_improvement": round(median_improvement, 4),
                            "better_count": better_count,
                            "total_count": total_count,
                            "proportion_better": round(better_count / total_count, 4) if total_count > 0 else 0.0,
                        })
            
            if comparison_results:
                comparison_df = pd.DataFrame(comparison_results)
                comparison_csv = output_path / "enrichment_reranking_comparison.csv"
                comparison_df.to_csv(comparison_csv, index=False, encoding="utf-8")
                print(f"[OK] Saved comparison to {comparison_csv}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_path}")
    print("\nAverage Scores by Configuration:")
    print("-" * 70)
    for _, row in stats.iterrows():
        config_name = row[group_by_col]
        print(f"\n{config_name.upper()}:")
        print(f"  Correctness:   {row['correctness_mean']:.4f} (±{row['correctness_std']:.4f})")
        print(f"  Faithfulness:  {row['faithfulness_mean']:.4f} (±{row['faithfulness_std']:.4f})")
        print(f"  Completeness:  {row['completeness_mean']:.4f} (±{row['completeness_std']:.4f})")
        print(f"  Conciseness:   {row['conciseness_mean']:.4f} (±{row['conciseness_std']:.4f})")
        print(f"  Overall:       {row['overall_mean']:.4f} (±{row['overall_std']:.4f})")
    
    # Check if all scores are zero (indicates a problem)
    all_zero = all(row['overall_mean'] == 0.0 for _, row in stats.iterrows())
    if all_zero:
        print("\n" + "!" * 70)
        print("WARNING: All scores are 0.0!")
        print("!" * 70)
        print("\nPossible causes:")
        print("1. RAG system is not finding relevant information (check index and retrieval)")
        print("2. LLM judge is failing to parse responses (check for JSON parsing warnings)")
        print("3. Index might not have data for the specified strategies")
        print("\nTo debug:")
        print("- Check the warnings above about RAG answers and chunk retrieval")
        print("- Verify the index has data: check Pinecone index contents")
        print("- Verify chunks are indexed with the correct 'chunking_strategy' metadata")
        print("- Run: python evaluation/test_llm_judge.py to verify judge is working")
        print("!" * 70)
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG answers using LLM-as-a-judge"
    )
    parser.add_argument(
        "--testset_file",
        type=str,
        default="tests/embedding_testset.json",
        help="Path to embedding_testset.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/llm_judge_results",
        help="Output directory for results",
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
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--api_keys_path",
        type=str,
        default=DEFAULT_API_KEYS_PATH,
        help="Path to API keys JSON file",
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default=None,
        help="Pinecone index name (default: from utils config)",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=DEFAULT_LLM_JUDGE_MODEL,
        help=f"Gemini model name for judging (default: {DEFAULT_LLM_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--test_enrichment_reranking",
        action="store_true",
        help="Test all 4 combinations of query enrichment and reranking",
    )
    parser.add_argument(
        "--baseline_strategy",
        type=str,
        default=None,
        help="Strategy to use as baseline for comparison. If not specified, uses first strategy in --strategies list. "
             "IMPORTANT: You should determine the best strategy/chunk_size/K from previous evaluations first.",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.testset_file):
        print(f"[ERROR] Testset file not found: {args.testset_file}")
        sys.exit(1)
    
    run_llm_judge_evaluation(
        testset_file=args.testset_file,
        output_dir=args.output_dir,
        strategies=args.strategies,
        top_k=args.top_k,
        api_keys_path=args.api_keys_path,
        index_name=args.index_name,
        gemini_model_name=args.judge_model,
        test_enrichment_reranking=args.test_enrichment_reranking,
        baseline_strategy=args.baseline_strategy,
    )


if __name__ == "__main__":
    main()
