"""
Evaluate Query Enrichment and Reranking for Any Configurations
================================================================

This script evaluates query enrichment and reranking (together and separately) 
for any chunk configurations and compares them with existing LLM judge statistics.

For each configuration, it can test various combinations:
- Baseline (no enrichment, no reranking) - optional
- Enrichment only
- Reranking only
- Both enrichment and reranking

Usage:
    # Using JSON config file
    python evaluation/evaluate_enrichment_reranking_configs.py \
        --configs_file evaluation/configs_to_test.json \
        --testset_file tests/embedding_testset.json \
        --output_dir evaluation/enrichment_reranking_results
    
    # Using command-line arguments
    python evaluation/evaluate_enrichment_reranking_configs.py \
        --chunk_config small_chunks --k 3 --strategy adaptive \
        --chunk_config small_chunks --k 10 --strategy adaptive \
        --testset_file tests/embedding_testset.json \
        --output_dir evaluation/enrichment_reranking_results
    
    # Include baseline in evaluation (don't skip it)
    python evaluation/evaluate_enrichment_reranking_configs.py \
        --configs_file evaluation/configs_to_test.json \
        --include_baseline \
        --testset_file tests/embedding_testset.json \
        --output_dir evaluation/enrichment_reranking_results

Config file format (JSON):
    [
        {"chunk_config": "small_chunks", "k": 3, "strategy": "adaptive", "index_name": "haifa-municipality-rag-small-chunks"},
        {"chunk_config": "small_overlap", "k": 10, "strategy": "adaptive"}
    ]
    
    Note: index_name is optional - if not provided, will be auto-generated from chunk_config.
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

from evaluation.run_llm_judge_evaluation import run_llm_judge_evaluation
from utils import DEFAULT_API_KEYS_PATH, DEFAULT_LLM_JUDGE_MODEL


def generate_index_name(chunk_config: str) -> str:
    """
    Generate Pinecone index name from chunk configuration name.
    
    Args:
        chunk_config: Chunk configuration name (e.g., "small_chunks", "small_overlap")
        
    Returns:
        Generated index name (e.g., "haifa-municipality-rag-small-chunks")
    """
    safe_config = chunk_config.replace("_", "-")
    return f"haifa-municipality-rag-{safe_config}"


def load_configurations(
    configs_file: Optional[str] = None,
    configs_args: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Load configurations from file or command-line arguments.
    
    Args:
        configs_file: Path to JSON file with configurations
        configs_args: List of configuration dicts from command-line
        
    Returns:
        List of configuration dictionaries with keys: chunk_config, k, strategy, index_name (optional)
    """
    configurations = []
    
    if configs_file and os.path.exists(configs_file):
        with open(configs_file, "r", encoding="utf-8") as f:
            configs_from_file = json.load(f)
            if isinstance(configs_from_file, list):
                configurations.extend(configs_from_file)
            else:
                raise ValueError(f"Config file must contain a JSON array, got {type(configs_from_file)}")
    
    if configs_args:
        configurations.extend(configs_args)
    
    # Validate and normalize configurations
    validated_configs = []
    for config in configurations:
        if not isinstance(config, dict):
            raise ValueError(f"Each configuration must be a dict, got {type(config)}")
        
        required_keys = ["chunk_config", "k", "strategy"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Configuration missing required keys: {missing_keys}")
        
        # Auto-generate index_name if not provided
        if "index_name" not in config or not config["index_name"]:
            config["index_name"] = generate_index_name(config["chunk_config"])
        
        validated_configs.append(config)
    
    return validated_configs


def load_combined_statistics(csv_path: str) -> pd.DataFrame:
    """Load the combined LLM judge statistics CSV."""
    if not os.path.exists(csv_path):
        print(f"[WARN] Combined statistics file not found: {csv_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"[OK] Loaded {len(df)} rows from combined statistics")
    return df


def run_evaluation_for_config(
    chunk_config: str,
    k: int,
    strategy: str,
    testset_file: str,
    output_base_dir: str,
    api_keys_path: str = DEFAULT_API_KEYS_PATH,
    index_name: Optional[str] = None,
    gemini_model_name: str = DEFAULT_LLM_JUDGE_MODEL,
    include_baseline: bool = False,
) -> Dict[str, Any]:
    """
    Run LLM judge evaluation with enrichment/reranking for a specific configuration.
    
    Args:
        chunk_config: Chunk configuration name (e.g., "small_chunks", "small_overlap")
        k: Number of chunks to retrieve
        strategy: Chunking strategy ("adaptive", "sentence", "baseline")
        testset_file: Path to testset JSON file
        output_base_dir: Base directory for output
        api_keys_path: Path to API keys file
        index_name: Pinecone index name (if None, auto-generated from chunk_config)
        gemini_model_name: Gemini model for judging
        include_baseline: If True, include baseline (no enrichment, no reranking) in evaluation
        
    Returns:
        Dictionary with configuration info and output directory path
    """
    # Determine index name
    if index_name is None:
        index_name = generate_index_name(chunk_config)
    
    # Create output directory
    config_folder = f"{chunk_config}_k{k}_{strategy}"
    output_dir = Path(output_base_dir) / config_folder
    
    print(f"\n{'=' * 70}")
    print(f"EVALUATING: {config_folder}")
    print(f"{'=' * 70}")
    print(f"Chunk Config: {chunk_config}")
    print(f"K: {k}")
    print(f"Strategy: {strategy}")
    print(f"Index: {index_name}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}\n")
    
    # Import the necessary functions to run custom evaluation
    from evaluation.run_llm_judge_evaluation import (
        load_testset,
        create_gold_answer_from_documents,
        evaluate_with_llm_judge,
    )
    from gemini_integration import GeminiRAG, init_gemini, load_api_keys
    from utils import DEFAULT_GEMINI_MODEL
    import pandas as pd
    from tqdm import tqdm
    
    # Load testset
    queries = load_testset(testset_file)
    
    # Initialize RAG system
    rag = GeminiRAG(
        api_keys_path=api_keys_path,
        index_name=index_name,
    )
    
    # Initialize Gemini model for judging
    api_keys = load_api_keys(api_keys_path)
    gemini_model = init_gemini(api_keys, gemini_model_name)
    
    # Define combinations to test
    combinations = []
    if include_baseline:
        combinations.append((False, False, "baseline"))
    combinations.extend([
        (True, False, "enrichment_only"),
        (False, True, "reranking_only"),
        (True, True, "both"),
    ])
    
    combo_count = len(combinations)
    baseline_note = "including baseline" if include_baseline else "excluding baseline"
    print(f"[EVALUATION] Testing {len(queries)} queries with {combo_count} combinations ({baseline_note})...")
    
    results = []
    for query_info in tqdm(queries, desc="Queries"):
        query = query_info.get("query") or query_info.get("question", "")
        documents = query_info.get("documents", [])
        
        # Create gold answer from relevant documents
        gold_answer = create_gold_answer_from_documents(documents)
        
        # Evaluate with each combination
        for use_enrich, use_rerank, combo_label in combinations:
            eval_result = evaluate_with_llm_judge(
                rag=rag,
                gemini_model=gemini_model,
                query=query,
                gold_answer=gold_answer,
                strategy=strategy,
                top_k=k,
                use_query_enhancement=use_enrich,
                use_reranking=use_rerank,
            )
            
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
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    results_csv = output_dir_path / "llm_judge_results.csv"
    df.to_csv(results_csv, index=False, encoding="utf-8")
    print(f"\n[OK] Saved results to {results_csv}")
    
    # Compute statistics
    stats = df.groupby("config").agg({
        "correctness": ["mean", "std"],
        "faithfulness": ["mean", "std"],
        "completeness": ["mean", "std"],
        "conciseness": ["mean", "std"],
        "overall": ["mean", "std"],
    }).round(4)
    
    stats.columns = ["_".join(col).strip() for col in stats.columns]
    stats = stats.reset_index()
    
    stats_csv = output_dir_path / "llm_judge_statistics.csv"
    
    # Add baseline from llm_judge_eval_results if not included in evaluation
    if not include_baseline:
        combined_stats_path = Path(__file__).parent / "llm_judge_eval_results" / "llm_judge" / "combined_llm_judge_statistics.csv"
        if combined_stats_path.exists():
            existing_stats = load_combined_statistics(str(combined_stats_path))
            
            # Find matching baseline
            baseline_match = existing_stats[
                (existing_stats["chunk_config"] == chunk_config) &
                (existing_stats["k"] == k) &
                (existing_stats["strategy"] == strategy)
            ]
            
            if len(baseline_match) > 0:
                baseline_row = baseline_match.iloc[0]
                
                # Create baseline stats row in the same format as our stats
                baseline_stats_row = {
                    "config": f"{strategy}_baseline",
                    "correctness_mean": baseline_row["correctness_mean"],
                    "correctness_std": baseline_row["correctness_std"],
                    "faithfulness_mean": baseline_row["faithfulness_mean"],
                    "faithfulness_std": baseline_row["faithfulness_std"],
                    "completeness_mean": baseline_row["completeness_mean"],
                    "completeness_std": baseline_row["completeness_std"],
                    "conciseness_mean": baseline_row["conciseness_mean"],
                    "conciseness_std": baseline_row["conciseness_std"],
                    "overall_mean": baseline_row["overall_mean"],
                    "overall_std": baseline_row["overall_std"],
                }
                
                # Convert to DataFrame and append to stats
                baseline_df = pd.DataFrame([baseline_stats_row])
                stats = pd.concat([baseline_df, stats], ignore_index=True)
                print(f"[OK] Added baseline from llm_judge_eval_results")
            else:
                print(f"[WARN] No baseline found in llm_judge_eval_results for {chunk_config} k{k} {strategy}")
        else:
            print(f"[WARN] Combined statistics file not found: {combined_stats_path}")
    
    stats.to_csv(stats_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved statistics to {stats_csv}")
    
    baseline_note = "with baseline included" if include_baseline else "(baseline added from llm_judge_eval_results)"
    print(f"\n[OK] Evaluation complete {baseline_note}")
    
    return {
        "chunk_config": chunk_config,
        "k": k,
        "strategy": strategy,
        "index_name": index_name,
        "output_dir": str(output_dir),
        "config_folder": config_folder,
    }


def aggregate_results(
    results: List[Dict[str, Any]],
    output_base_dir: str,
    combined_stats_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregate results from all configurations and compare with combined statistics.
    
    Args:
        results: List of configuration results dictionaries
        output_base_dir: Base output directory
        combined_stats_path: Path to combined_llm_judge_statistics.csv
        
    Returns:
        DataFrame with aggregated results
    """
    all_stats = []
    
    # Load results from each configuration
    for config_result in results:
        output_dir = Path(config_result["output_dir"])
        stats_csv = output_dir / "llm_judge_statistics.csv"
        
        if not stats_csv.exists():
            print(f"[WARN] Statistics file not found: {stats_csv}")
            continue
        
        df = pd.read_csv(stats_csv, encoding="utf-8")
        
        # Add configuration metadata
        df["chunk_config"] = config_result["chunk_config"]
        df["k"] = config_result["k"]
        df["strategy"] = config_result["strategy"]
        df["config_folder"] = config_result["config_folder"]
        
        all_stats.append(df)
    
    if not all_stats:
        print("[ERROR] No statistics found to aggregate")
        return pd.DataFrame()
    
    # Combine all statistics
    combined_df = pd.concat(all_stats, ignore_index=True)
    
    # Add baselines from llm_judge_eval_results to aggregated results
    if combined_stats_path and os.path.exists(combined_stats_path):
        existing_stats = load_combined_statistics(combined_stats_path)
        
        if not existing_stats.empty:
            baseline_rows = []
            for config_result in results:
                chunk_config = config_result["chunk_config"]
                k = config_result["k"]
                strategy = config_result["strategy"]
                
                # Find matching baseline
                baseline_match = existing_stats[
                    (existing_stats["chunk_config"] == chunk_config) &
                    (existing_stats["k"] == k) &
                    (existing_stats["strategy"] == strategy)
                ]
                
                if len(baseline_match) > 0:
                    baseline_row = baseline_match.iloc[0]
                    
                    # Create baseline row in same format
                    baseline_stats_row = {
                        "config": f"{strategy}_baseline",
                        "chunk_config": chunk_config,
                        "k": k,
                        "strategy": strategy,
                        "config_folder": config_result["config_folder"],
                        "correctness_mean": baseline_row["correctness_mean"],
                        "correctness_std": baseline_row["correctness_std"],
                        "faithfulness_mean": baseline_row["faithfulness_mean"],
                        "faithfulness_std": baseline_row["faithfulness_std"],
                        "completeness_mean": baseline_row["completeness_mean"],
                        "completeness_std": baseline_row["completeness_std"],
                        "conciseness_mean": baseline_row["conciseness_mean"],
                        "conciseness_std": baseline_row["conciseness_std"],
                        "overall_mean": baseline_row["overall_mean"],
                        "overall_std": baseline_row["overall_std"],
                    }
                    baseline_rows.append(baseline_stats_row)
            
            if baseline_rows:
                baseline_df = pd.DataFrame(baseline_rows)
                # Combine with existing results
                combined_df = pd.concat([baseline_df, combined_df], ignore_index=True)
                # Sort to ensure baselines come first for each configuration
                # Create a sort key: 0 for baseline, 1 for others
                combined_df["_sort_key"] = combined_df["config"].apply(
                    lambda x: 0 if "baseline" in str(x).lower() else 1
                )
                combined_df = combined_df.sort_values(
                    by=["chunk_config", "k", "strategy", "_sort_key", "config"]
                ).drop(columns=["_sort_key"]).reset_index(drop=True)
                print(f"[OK] Added {len(baseline_rows)} baseline(s) from llm_judge_eval_results to aggregated results")
    
    # Save aggregated results
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    aggregated_csv = output_base / "aggregated_enrichment_reranking_statistics.csv"
    combined_df.to_csv(aggregated_csv, index=False, encoding="utf-8")
    print(f"\n[OK] Saved aggregated statistics to {aggregated_csv}")
    
    # Compare with existing combined statistics if provided
    if combined_stats_path and os.path.exists(combined_stats_path):
        print(f"\n[STEP] Comparing with existing statistics: {combined_stats_path}")
        existing_stats = load_combined_statistics(combined_stats_path)
        
        if not existing_stats.empty:
            # Create comparison
            comparison_results = []
            
            # For each configuration in our results, find matching baseline in existing stats
            for _, row in combined_df.iterrows():
                chunk_config = row["chunk_config"]
                k = row["k"]
                strategy = row["strategy"]
                config_name = row["config"]  # This includes enrichment/reranking info
                
                # Skip baseline configs in our results (we're comparing against existing baseline)
                if "baseline" in config_name.lower():
                    continue
                
                # Find matching baseline in existing stats
                baseline_match = existing_stats[
                    (existing_stats["chunk_config"] == chunk_config) &
                    (existing_stats["k"] == k) &
                    (existing_stats["strategy"] == strategy)
                ]
                
                if len(baseline_match) > 0:
                    baseline_row = baseline_match.iloc[0]
                    
                    # Compare metrics
                    metrics = ["correctness", "faithfulness", "completeness", "conciseness", "overall"]
                    for metric in metrics:
                        mean_col = f"{metric}_mean"
                        std_col = f"{metric}_std"
                        
                        if mean_col in row and mean_col in baseline_row:
                            new_mean = row[mean_col]
                            baseline_mean = baseline_row[mean_col]
                            
                            if pd.notna(new_mean) and pd.notna(baseline_mean):
                                improvement = new_mean - baseline_mean
                                improvement_pct = (improvement / baseline_mean * 100) if baseline_mean > 0 else 0
                                
                                comparison_results.append({
                                    "chunk_config": chunk_config,
                                    "k": k,
                                    "strategy": strategy,
                                    "config": config_name,
                                    "metric": metric,
                                    "baseline_mean": baseline_mean,
                                    "new_mean": new_mean,
                                    "improvement": improvement,
                                    "improvement_pct": improvement_pct,
                                    "baseline_std": baseline_row.get(std_col, 0),
                                    "new_std": row.get(std_col, 0),
                                })
            
            if comparison_results:
                comparison_df = pd.DataFrame(comparison_results)
                comparison_csv = output_base / "enrichment_reranking_vs_baseline_comparison.csv"
                comparison_df.to_csv(comparison_csv, index=False, encoding="utf-8")
                print(f"[OK] Saved comparison to {comparison_csv}")
                
                # Print summary
                print("\n" + "=" * 70)
                print("COMPARISON SUMMARY")
                print("=" * 70)
                
                # Group by config and show best improvements
                for config in comparison_df["config"].unique():
                    config_comparison = comparison_df[comparison_df["config"] == config]
                    print(f"\n{config.upper()}:")
                    for metric in ["overall", "correctness", "faithfulness", "completeness"]:
                        metric_data = config_comparison[config_comparison["metric"] == metric]
                        if len(metric_data) > 0:
                            row = metric_data.iloc[0]
                            print(f"  {metric}: {row['new_mean']:.4f} vs {row['baseline_mean']:.4f} "
                                  f"({row['improvement_pct']:+.2f}%)")
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate query enrichment and reranking for any configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using JSON config file
  python evaluation/evaluate_enrichment_reranking_configs.py \\
      --configs_file evaluation/configs_to_test.json \\
      --testset_file tests/embedding_testset.json

  # Using command-line arguments (multiple configs)
  python evaluation/evaluate_enrichment_reranking_configs.py \\
      --chunk_config small_chunks --k 3 --strategy adaptive \\
      --chunk_config small_chunks --k 10 --strategy adaptive \\
      --testset_file tests/embedding_testset.json

  # Include baseline in evaluation
  python evaluation/evaluate_enrichment_reranking_configs.py \\
      --configs_file evaluation/configs_to_test.json \\
      --include_baseline \\
      --testset_file tests/embedding_testset.json
        """
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
        default="evaluation/enrichment_reranking_results",
        help="Base output directory for results",
    )
    parser.add_argument(
        "--api_keys_path",
        type=str,
        default=DEFAULT_API_KEYS_PATH,
        help="Path to API keys JSON file",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=DEFAULT_LLM_JUDGE_MODEL,
        help=f"Gemini model name for judging (default: {DEFAULT_LLM_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--combined_stats_path",
        type=str,
        default="evaluation/llm_judge_eval_results/llm_judge/combined_llm_judge_statistics.csv",
        help="Path to combined_llm_judge_statistics.csv for comparison",
    )
    parser.add_argument(
        "--configs_file",
        type=str,
        default=None,
        help="JSON file with list of configurations to test (optional)",
    )
    parser.add_argument(
        "--chunk_config",
        type=str,
        action="append",
        default=None,
        help="Chunk configuration name (can be specified multiple times)",
    )
    parser.add_argument(
        "--k",
        type=int,
        action="append",
        default=None,
        help="Number of chunks to retrieve (can be specified multiple times, must match --chunk_config count)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        action="append",
        default=None,
        help="Chunking strategy: adaptive, sentence, or baseline (can be specified multiple times, must match --chunk_config count)",
    )
    parser.add_argument(
        "--index_name",
        type=str,
        action="append",
        default=None,
        help="Pinecone index name (optional, can be specified multiple times, auto-generated if not provided)",
    )
    parser.add_argument(
        "--include_baseline",
        action="store_true",
        help="Include baseline (no enrichment, no reranking) in evaluation. If False, baseline will be loaded from llm_judge_eval_results for comparison.",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.testset_file):
        print(f"[ERROR] Testset file not found: {args.testset_file}")
        sys.exit(1)
    
    # Load configurations
    configs_from_args = None
    if args.chunk_config:
        if not args.k or not args.strategy:
            print("[ERROR] --chunk_config requires --k and --strategy")
            sys.exit(1)
        
        if len(args.chunk_config) != len(args.k) or len(args.chunk_config) != len(args.strategy):
            print("[ERROR] --chunk_config, --k, and --strategy must have the same number of values")
            sys.exit(1)
        
        configs_from_args = []
        for i, chunk_config in enumerate(args.chunk_config):
            config = {
                "chunk_config": chunk_config,
                "k": args.k[i],
                "strategy": args.strategy[i],
            }
            if args.index_name and i < len(args.index_name):
                config["index_name"] = args.index_name[i]
            configs_from_args.append(config)
    
    try:
        configurations = load_configurations(
            configs_file=args.configs_file,
            configs_args=configs_from_args,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load configurations: {e}")
        sys.exit(1)
    
    if not configurations:
        print("[ERROR] No configurations specified. Use --configs_file or --chunk_config/--k/--strategy")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("ENRICHMENT & RERANKING EVALUATION")
    print("=" * 70)
    print(f"\nConfigurations to test: {len(configurations)}")
    for config in configurations:
        print(f"  - {config['chunk_config']} k{config['k']} {config['strategy']} (index: {config['index_name']})")
    print(f"\nTestset: {args.testset_file}")
    print(f"Output: {args.output_dir}")
    if args.include_baseline:
        print("\nNOTE: Baseline (no enrichment, no reranking) will be included in evaluation.")
    else:
        print("\nNOTE: Baseline (no enrichment, no reranking) will be skipped.")
        print("      Baseline comparison will use existing data from llm_judge_eval_results.")
    print("=" * 70)
    
    # Run evaluation for each configuration
    results = []
    for config in configurations:
        try:
            result = run_evaluation_for_config(
                chunk_config=config["chunk_config"],
                k=config["k"],
                strategy=config["strategy"],
                testset_file=args.testset_file,
                output_base_dir=args.output_dir,
                api_keys_path=args.api_keys_path,
                index_name=config.get("index_name"),
                gemini_model_name=args.judge_model,
                include_baseline=args.include_baseline,
            )
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed to evaluate {config['chunk_config']} k{config['k']} {config['strategy']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Aggregate results and compare
    if results:
        print("\n" + "=" * 70)
        print("AGGREGATING RESULTS")
        print("=" * 70)
        aggregated_df = aggregate_results(
            results=results,
            output_base_dir=args.output_dir,
            combined_stats_path=args.combined_stats_path,
        )
        
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)
        print(f"\nResults saved to: {args.output_dir}")
        print(f"Total configurations evaluated: {len(results)}")
    else:
        print("\n[ERROR] No configurations were successfully evaluated")
        sys.exit(1)


if __name__ == "__main__":
    main()
