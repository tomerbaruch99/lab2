"""
Evaluate Different Chunk Size and Overlap Configurations
=========================================================

This script helps you evaluate different chunk size and overlap configurations.
It automates the workflow of:
1. Preparing data with different chunk configurations
2. Indexing the data
3. Running evaluation
4. Comparing results

Usage:
    python evaluation/evaluate_chunk_configurations.py \
        --input_json scrape_and_prepare_data/haifa_scraped_data.json \
        --configs configs.json \
        --output_base_dir ./chunk_config_evaluations

Config file format (configs.json):
    [
        {
            "name": "small_chunks",
            "chunk_chars": 500,
            "chunk_overlap": 100
        },
        {
            "name": "medium_chunks",
            "chunk_chars": 1000,
            "chunk_overlap": 200
        },
        {
            "name": "large_chunks",
            "chunk_chars": 2000,
            "chunk_overlap": 400
        }
    ]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.generate_evaluation_results import run_evaluation, load_testset_with_ground_truth, DEFAULT_QUERIES


def prepare_data_for_config(
    input_json: str,
    output_dir: str,
    chunk_chars: int,
    chunk_overlap: int,
    config_name: str,
) -> str:
    """
    Prepare data for a specific chunk configuration.
    
    Returns:
        Path to the prepared parquet file
    """
    print(f"\n{'='*70}")
    print(f"Preparing data for config: {config_name}")
    print(f"  chunk_chars: {chunk_chars}, chunk_overlap: {chunk_overlap}")
    print(f"{'='*70}\n")
    
    config_output_dir = Path(output_dir) / config_name
    config_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run data preparation
    prep_script = Path(__file__).parent.parent / "scrape_and_prepare_data" / "data_preparation.py"
    
    cmd = [
        sys.executable,
        str(prep_script),
        "--input_json", input_json,
        "--out_dir", str(config_output_dir),
        "--chunk_chars", str(chunk_chars),
        "--chunk_overlap", str(chunk_overlap),
    ]
    
    print(f"[STEP] Running data preparation...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Data preparation failed:")
        print(result.stderr)
        raise RuntimeError(f"Data preparation failed for {config_name}")
    
    # Find the generated parquet file
    parquet_files = list(config_output_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet file generated in {config_output_dir}")
    
    parquet_file = parquet_files[0]
    print(f"[OK] Data prepared: {parquet_file}")
    return str(parquet_file)


def index_data(
    parquet_file: str,
    index_name: str,
    api_keys_path: str,
) -> None:
    """
    Index the prepared data into Pinecone.
    """
    print(f"\n[STEP] Indexing data to Pinecone index: {index_name}...")
    
    indexing_script = Path(__file__).parent.parent / "indexing.py"
    
    cmd = [
        sys.executable,
        str(indexing_script),
        "--prepared_file", parquet_file,
        "--api_keys_path", api_keys_path,
        "--index_name", index_name,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Indexing failed:")
        print(result.stderr)
        raise RuntimeError(f"Indexing failed for {index_name}")
    
    print(f"[OK] Data indexed to {index_name}")


def run_evaluation_for_config(
    config_name: str,
    index_name: str,
    output_dir: str,
    queries: List[Dict],
    api_keys_path: str,
    strategies: List[str] = None,
    top_k: int = 5,
) -> str:
    """
    Run evaluation for a specific configuration.
    
    Returns:
        Path to evaluation results directory
    """
    if strategies is None:
        strategies = ["baseline", "sentence", "adaptive"]
    
    eval_output_dir = Path(output_dir) / config_name / "evaluation_results"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[STEP] Running evaluation for {config_name}...")
    print(f"       Using index: {index_name}")
    
    # Run evaluation with the specific index name
    run_evaluation(
        queries=queries,
        output_dir=str(eval_output_dir),
        strategies=strategies,
        top_k=top_k,
        api_keys_path=api_keys_path,
        include_baselines=False,
        index_name=index_name,
    )
    
    print(f"[OK] Evaluation complete: {eval_output_dir}")
    return str(eval_output_dir)


def load_configs(config_file: str) -> List[Dict]:
    """Load chunk configurations from JSON file."""
    with open(config_file, "r", encoding="utf-8") as f:
        configs = json.load(f)
    
    # Validate configs
    for config in configs:
        required = ["name", "chunk_chars", "chunk_overlap"]
        for field in required:
            if field not in config:
                raise ValueError(f"Config missing required field: {field}")
    
    return configs


def compare_config_results(results_dir: str) -> None:
    """
    Compare results across different configurations.
    Creates a summary comparison CSV.
    """
    results_path = Path(results_dir)
    config_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    
    all_stats = []
    for config_dir in config_dirs:
        stats_file = config_dir / "evaluation_results" / "strategy_statistics.csv"
        if stats_file.exists():
            import pandas as pd
            df = pd.read_csv(stats_file)
            df["config_name"] = config_dir.name
            all_stats.append(df)
    
    if all_stats:
        import pandas as pd
        combined = pd.concat(all_stats, ignore_index=True)
        comparison_file = results_path / "config_comparison.csv"
        combined.to_csv(comparison_file, index=False, encoding="utf-8")
        print(f"\n[OK] Configuration comparison saved to: {comparison_file}")
        print("\nSummary:")
        print(combined.groupby(["config_name", "strategy"])[["avg_score_mean", "namespace_correct_mean"]].mean())


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate different chunk size and overlap configurations (tests multiple configs with API access)"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON file (scraped data)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        required=True,
        help="JSON file with chunk configurations to test",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="./chunk_config_evaluations",
        help="Base directory for all configuration outputs",
    )
    parser.add_argument(
        "--api_keys_path",
        type=str,
        default="../utils/api_keys.json",
        help="Path to API keys JSON file",
    )
    parser.add_argument(
        "--testset_file",
        type=str,
        default=None,
        help="Path to testset with ground truth (tests/embedding_testset.json)",
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
        "--skip_preparation",
        action="store_true",
        help="Skip data preparation (use existing prepared files)",
    )
    parser.add_argument(
        "--skip_indexing",
        action="store_true",
        help="Skip indexing (use existing indexes)",
    )
    
    args = parser.parse_args()
    
    # Load configurations
    configs = load_configs(args.configs)
    print(f"\n[INFO] Loaded {len(configs)} configurations to test")
    
    # Load queries
    if args.testset_file and os.path.exists(args.testset_file):
        queries = load_testset_with_ground_truth(args.testset_file)
        print(f"[INFO] Using testset with ground truth: {args.testset_file}")
    else:
        # Try default testset
        default_testset = Path(__file__).parent.parent / "tests" / "embedding_testset.json"
        if default_testset.exists():
            queries = load_testset_with_ground_truth(str(default_testset))
            print(f"[INFO] Using default testset: {default_testset}")
        else:
            queries = DEFAULT_QUERIES
            print(f"[INFO] Using default queries")
    
    output_base = Path(args.output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Process each configuration
    for config in configs:
        config_name = config["name"]
        chunk_chars = config["chunk_chars"]
        chunk_overlap = config["chunk_overlap"]
        index_name = f"haifa-municipality-rag-{config_name}"
        
        try:
            # Step 1: Prepare data
            if not args.skip_preparation:
                parquet_file = prepare_data_for_config(
                    input_json=args.input_json,
                    output_dir=str(output_base),
                    chunk_chars=chunk_chars,
                    chunk_overlap=chunk_overlap,
                    config_name=config_name,
                )
            else:
                # Find existing parquet file
                config_dir = output_base / config_name
                parquet_files = list(config_dir.glob("**/*.parquet"))
                if not parquet_files:
                    raise FileNotFoundError(f"No parquet file found for {config_name}")
                parquet_file = str(parquet_files[0])
                print(f"[INFO] Using existing prepared file: {parquet_file}")
            
            # Step 2: Index data
            if not args.skip_indexing:
                index_data(
                    parquet_file=parquet_file,
                    index_name=index_name,
                    api_keys_path=args.api_keys_path,
                )
            else:
                print(f"[INFO] Skipping indexing, using existing index: {index_name}")
            
            # Step 3: Run evaluation
            eval_dir = run_evaluation_for_config(
                config_name=config_name,
                index_name=index_name,
                output_dir=str(output_base),
                queries=queries,
                api_keys_path=args.api_keys_path,
                strategies=args.strategies,
                top_k=args.top_k,
            )
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process configuration {config_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare results across configurations
    print(f"\n{'='*70}")
    print("COMPARING CONFIGURATIONS")
    print(f"{'='*70}")
    try:
        compare_config_results(str(output_base))
    except Exception as e:
        print(f"[WARN] Could not generate comparison: {e}")
        print("       You can manually compare the CSV files in each config directory")
    
    print(f"\n{'='*70}")
    print("CONFIGURATION EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_base}")
    print("\nNext steps:")
    print("1. Review individual evaluation results in each config subdirectory")
    print("2. Compare results using the evaluation notebook")
    print("3. Check config_comparison.csv for summary statistics")


if __name__ == "__main__":
    main()


