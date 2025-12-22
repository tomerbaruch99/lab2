"""
Comprehensive Strategy Comparison Script
========================================

This script runs a comprehensive evaluation across all combinations:
- 3 chunk configurations (small_chunks, medium_chunks, small_overlap)
- 3 K values (3, 5, 10)
- 3 strategies (baseline, sentence, adaptive)

It generates:
1. LLM judge results for each combination (27 total)
2. Baseline comparison results for each chunk config/K value (9 total)
3. Summary report aggregating all results

This script does NOT re-index data - it uses existing Pinecone indexes.

Usage:
    python evaluation/run_comprehensive_comparison.py
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import DEFAULT_API_KEYS_PATH


def run_llm_judge_evaluation(
    testset_file: str,
    output_dir: str,
    strategy: str,
    top_k: int,
    index_name: str,
    api_keys_path: str = DEFAULT_API_KEYS_PATH,
) -> bool:
    """Run LLM judge evaluation for a single combination."""
    cmd = [
        sys.executable,
        "evaluation/run_llm_judge_evaluation.py",
        "--testset_file", testset_file,
        "--output_dir", output_dir,
        "--strategies", strategy,
        "--top_k", str(top_k),
        "--index_name", index_name,
        "--api_keys_path", api_keys_path,
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"      [ERROR] Failed: {e.stderr}", file=sys.stderr)
        return False


def run_baseline_evaluation(
    testset_file: str,
    output_dir: str,
    strategies: List[str],
    top_k: int,
    index_name: str,
    api_keys_path: str = DEFAULT_API_KEYS_PATH,
) -> bool:
    """Run baseline evaluation for a chunk config/K value combination."""
    cmd = [
        sys.executable,
        "evaluation/generate_evaluation_results.py",
        "--strategies"] + strategies + [
        "--top_k", str(top_k),
        "--include_baselines",
        "--testset_file", testset_file,
        "--output_dir", output_dir,
        "--index_name", index_name,
        "--api_keys_path", api_keys_path,
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"    [ERROR] Failed: {e.stderr}", file=sys.stderr)
        return False


def generate_summary(base_output_dir: Path):
    """Generate summary report from all results."""
    import pandas as pd
    
    results_summary = []
    
    print("\nCollecting results...")
    
    # Collect LLM judge results
    llm_judge_dir = base_output_dir / "llm_judge"
    if llm_judge_dir.exists():
        for chunk_config in ["small_chunks", "medium_chunks", "small_overlap"]:
            for k in [3, 5, 10]:
                for strategy in ["baseline", "sentence", "adaptive"]:
                    stats_file = llm_judge_dir / f"{chunk_config}_k{k}_{strategy}" / "llm_judge_statistics.csv"
                    if stats_file.exists():
                        try:
                            df = pd.read_csv(stats_file)
                            for _, row in df.iterrows():
                                strategy_name = row.get("strategy", strategy)
                                results_summary.append({
                                    "chunk_config": chunk_config,
                                    "k": k,
                                    "evaluation_type": "llm_judge",
                                    "strategy": strategy_name,
                                    "correctness_mean": row.get("correctness_mean", 0),
                                    "faithfulness_mean": row.get("faithfulness_mean", 0),
                                    "completeness_mean": row.get("completeness_mean", 0),
                                    "conciseness_mean": row.get("conciseness_mean", 0),
                                    "overall_mean": row.get("overall_mean", 0),
                                })
                        except Exception as e:
                            print(f"Error reading {stats_file}: {e}")
    
    # Collect baseline results
    baselines_dir = base_output_dir / "baselines"
    if baselines_dir.exists():
        for chunk_config in ["small_chunks", "medium_chunks", "small_overlap"]:
            for k in [3, 5, 10]:
                stats_file = baselines_dir / f"{chunk_config}_k{k}_baselines" / "strategy_statistics.csv"
                if stats_file.exists():
                    try:
                        df = pd.read_csv(stats_file)
                        for _, row in df.iterrows():
                            results_summary.append({
                                "chunk_config": chunk_config,
                                "k": k,
                                "evaluation_type": "retrieval_metrics",
                                "strategy": row["strategy"],
                                "avg_score_mean": row.get("avg_score_mean", 0),
                                "namespace_correct_mean": row.get("namespace_correct_mean", 0),
                                "precision_mean": row.get("precision_mean", None),
                                "recall_mean": row.get("recall_mean", None),
                            })
                    except Exception as e:
                        print(f"Error reading {stats_file}: {e}")
    
    # Save summary
    if len(results_summary) > 0:
        summary_df = pd.DataFrame(results_summary)
        summary_file = base_output_dir / "comparison_summary.csv"
        summary_df.to_csv(summary_file, index=False, encoding="utf-8")
        print(f"\nSummary saved to: {summary_file}")
        print(f"Total results collected: {len(results_summary)}")
        
        # Print top performers
        print("\n=== Top 10 Performers by Overall Score (LLM Judge) ===")
        llm_results = summary_df[summary_df["evaluation_type"] == "llm_judge"]
        if len(llm_results) > 0:
            top_overall = llm_results.nlargest(10, "overall_mean")
            print(top_overall[["chunk_config", "k", "strategy", "overall_mean", "correctness_mean", "faithfulness_mean"]].to_string(index=False))
        else:
            print("No LLM judge results found.")
        
        # Print baseline comparison
        print("\n=== Baseline Comparison (Retrieval Metrics) ===")
        baseline_results = summary_df[summary_df["evaluation_type"] == "retrieval_metrics"]
        if len(baseline_results) > 0:
            baseline_summary = baseline_results.groupby("strategy").agg({
                "avg_score_mean": "mean",
                "namespace_correct_mean": "mean",
                "precision_mean": "mean",
                "recall_mean": "mean"
            }).round(4)
            print(baseline_summary.to_string())
        else:
            print("No baseline results found.")
    else:
        print("No results found to summarize.")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive strategy comparison across all chunk configs, K values, and strategies"
    )
    parser.add_argument(
        "--testset_file",
        type=str,
        default="tests/embedding_testset.json",
        help="Path to testset JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/comprehensive_comparison_results",
        help="Base output directory for results",
    )
    parser.add_argument(
        "--api_keys_path",
        type=str,
        default=DEFAULT_API_KEYS_PATH,
        help="Path to API keys JSON file",
    )
    
    args = parser.parse_args()
    
    # Define configurations
    chunk_configs = [
        {"name": "small_chunks", "index": "haifa-municipality-rag-small-chunks"},
        {"name": "medium_chunks", "index": "haifa-municipality-rag-medium-chunks"},
        {"name": "small_overlap", "index": "haifa-municipality-rag-small-overlap"},
    ]
    
    k_values = [3, 5, 10]
    strategies = ["baseline", "sentence", "adaptive"]
    
    # Create output directories
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    llm_judge_dir = base_output_dir / "llm_judge"
    baselines_dir = base_output_dir / "baselines"
    llm_judge_dir.mkdir(parents=True, exist_ok=True)
    baselines_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Comprehensive Strategy Comparison")
    print("Using LLM Judge Evaluation")
    print("=" * 70)
    print()
    print("Running each combination separately:")
    print("  - 3 chunk configs × 3 k values × 3 strategies = 27 LLM judge evaluations")
    print("  - 3 chunk configs × 3 k values = 9 baseline evaluations")
    print()
    
    total_runs = len(chunk_configs) * len(k_values) * len(strategies)
    current_run = 0
    skipped_count = 0
    completed_count = 0
    
    # Run LLM judge evaluations
    for chunk_config in chunk_configs:
        chunk_name = chunk_config["name"]
        index_name = chunk_config["index"]
        
        print("-" * 70)
        print(f"Chunk Config: {chunk_name}")
        print(f"Index: {index_name}")
        print("-" * 70)
        print()
        
        for k in k_values:
            print(f"  Testing k={k}...")
            
            for strategy in strategies:
                current_run += 1
                
                # Create output directory for this combination
                output_dir = llm_judge_dir / f"{chunk_name}_k{k}_{strategy}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Check if already completed
                results_file = output_dir / "llm_judge_statistics.csv"
                
                if results_file.exists():
                    skipped_count += 1
                    print(f"    [{current_run}/{total_runs}] {chunk_name} / k={k} / strategy={strategy} [SKIPPED - Already completed]")
                    continue
                
                print(f"    [{current_run}/{total_runs}] {chunk_name} / k={k} / strategy={strategy}")
                
                # Run LLM judge for this one strategy
                success = run_llm_judge_evaluation(
                    testset_file=args.testset_file,
                    output_dir=str(output_dir),
                    strategy=strategy,
                    top_k=k,
                    index_name=index_name,
                    api_keys_path=args.api_keys_path,
                )
                
                if success:
                    completed_count += 1
                    print(f"      [OK] Completed {chunk_name} k={k} strategy={strategy}")
                else:
                    print(f"      [ERROR] Failed for {chunk_name} k={k} strategy={strategy}")
            
            print()
    
    print()
    print(f"LLM Judge Summary: {completed_count} completed, {skipped_count} skipped")
    print()
    
    # Run baseline evaluations
    print("=" * 70)
    print("Running Baseline Evaluations")
    print("=" * 70)
    print()
    
    baseline_skipped_count = 0
    baseline_completed_count = 0
    
    for chunk_config in chunk_configs:
        chunk_name = chunk_config["name"]
        index_name = chunk_config["index"]
        
        print("-" * 70)
        print(f"Baseline Evaluations: {chunk_name}")
        print("-" * 70)
        print()
        
        for k in k_values:
            print(f"  Testing k={k} with baselines...")
            
            # Create output directory for baseline results
            baseline_output_dir = baselines_dir / f"{chunk_name}_k{k}_baselines"
            baseline_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if already completed
            baseline_results_file = baseline_output_dir / "strategy_statistics.csv"
            
            if baseline_results_file.exists():
                baseline_skipped_count += 1
                print("    [SKIPPED - Already completed]")
                print()
                continue
            
            print("    Running evaluation with baselines...")
            
            success = run_baseline_evaluation(
                testset_file=args.testset_file,
                output_dir=str(baseline_output_dir),
                strategies=strategies,
                top_k=k,
                index_name=index_name,
                api_keys_path=args.api_keys_path,
            )
            
            if success:
                baseline_completed_count += 1
                print(f"    [OK] Completed baseline evaluation for {chunk_name} k={k}")
            else:
                print(f"    [ERROR] Failed to run baseline evaluation for {chunk_name} k={k}")
                print("    Continuing with next configuration...")
            
            print()
    
    print()
    print(f"Baseline Summary: {baseline_completed_count} completed, {baseline_skipped_count} skipped")
    print()
    
    # Generate summary
    print("=" * 70)
    print("Generating Summary Report")
    print("=" * 70)
    print()
    
    try:
        generate_summary(base_output_dir)
        print("\n[OK] Summary generated successfully")
    except Exception as e:
        print(f"\n[WARN] Summary generation had issues: {e}")
        print("       Individual results are still available")
    
    print()
    print("=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    print()
    print(f"Results saved to: {base_output_dir}")
    print()
    print("Directory structure:")
    print(f"  {base_output_dir}/")
    print("    ├── llm_judge/")
    print("    │   ├── {{chunk_config}}_k{{k}}_{{strategy}}/     - LLM judge results for EACH combination")
    print("    │   └── (27 folders total)")
    print("    ├── baselines/")
    print("    │   ├── {{chunk_config}}_k{{k}}_baselines/       - Baseline comparison results")
    print("    │   └── (9 folders total)")
    print("    └── comparison_summary.csv                  - Aggregated summary of all results")
    print()
    print("Total combinations:")
    print(f"  - LLM Judge: {completed_count} completed, {skipped_count} skipped (out of 27 total)")
    print(f"  - Baselines: {baseline_completed_count} completed, {baseline_skipped_count} skipped (out of 9 total)")
    print()
    print("Note: Skipped combinations already have results files. Delete them to re-run.")


if __name__ == "__main__":
    main()

