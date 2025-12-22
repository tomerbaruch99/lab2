"""
Evaluate Different Chunk Size and Overlap Configurations
=========================================================

This script automates the evaluation of different chunk size and overlap configurations
for RAG (Retrieval-Augmented Generation) systems. It systematically tests multiple
chunking strategies to help identify optimal parameters for document chunking.

The script performs a complete evaluation pipeline:
1. Data Preparation: Processes input JSON data with specified chunk sizes and overlaps
2. Indexing: Creates Pinecone vector indexes for each configuration
3. Evaluation: Runs retrieval evaluations using multiple strategies (baseline, sentence, adaptive)
4. Comparison: Generates summary statistics comparing all configurations

This is particularly useful for:
- Finding optimal chunk sizes for your document corpus
- Understanding the trade-offs between chunk size and overlap
- Comparing retrieval performance across different chunking strategies

Usage:
    python evaluation/evaluate_chunk_configurations.py \
        --input_json scrape_and_prepare_data/haifa_scraped.json \
        --configs configs.json \
        --output_base_dir evaluation/chunks_config_comparison_eval_results

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

Output Structure:
    output_base_dir/
        config_name_1/
            prepared_data.parquet
            evaluation_results/
                strategy_statistics.csv
                ...
        config_name_2/
            ...
        config_comparison.csv  # Summary comparison across all configs
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to Python path to enable imports from project root
# This allows importing modules from the main project directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import evaluation functions and default queries from the evaluation module
from evaluation.generate_evaluation_results import (
    run_evaluation,
    load_testset_with_ground_truth,
    DEFAULT_QUERIES,
)
from utils import (
    DEFAULT_EVALUATION_STRATEGIES,
    DEFAULT_EVALUATION_TOP_K,
    DEFAULT_API_KEYS_PATH,
)


def prepare_data_for_config(
    input_json: str,
    output_dir: str,
    chunk_chars: int,
    chunk_overlap: int,
    config_name: str,
) -> str:
    """
    Prepare data for a specific chunk configuration by running the data preparation script.
    
    This function calls the data_preparation.py script with specific chunk size and overlap
    parameters. It creates a subdirectory for the configuration and processes the input JSON
    file into a parquet format suitable for indexing.
    
    Args:
        input_json: Path to the input JSON file containing scraped document data
        output_dir: Base directory where prepared data will be stored
        chunk_chars: Maximum number of characters per chunk
        chunk_overlap: Number of characters to overlap between consecutive chunks
        config_name: Unique name identifier for this configuration (used for subdirectory)
    
    Returns:
        str: Absolute path to the generated parquet file containing the prepared data
    
    Raises:
        RuntimeError: If the data preparation subprocess fails
        FileNotFoundError: If no parquet file is generated after preparation
    
    Note:
        The function creates a subdirectory structure: output_dir/config_name/
        The parquet file will be located within this subdirectory.
    """
    # Display configuration information for this data preparation run
    print(f"\n{'='*70}")
    print(f"Preparing data for config: {config_name}")
    print(f"  chunk_chars: {chunk_chars}, chunk_overlap: {chunk_overlap}")
    print(f"{'='*70}\n")
    
    # Create output directory structure for this specific configuration
    config_output_dir = Path(output_dir) / config_name
    config_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command to run the data preparation script as a subprocess
    # This allows us to reuse the existing data preparation logic with different parameters
    cmd = [
        sys.executable,
        "scrape_and_prepare_data/data_preparation.py",
        "--input_json", input_json,
        "--out_dir", str(config_output_dir),
        "--chunk_chars", str(chunk_chars),
        "--chunk_overlap", str(chunk_overlap),
    ]
    
    print(f"[STEP] Running data preparation...")
    # Execute the data preparation script and capture both stdout and stderr
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check if the subprocess completed successfully
    if result.returncode != 0:
        print(f"[ERROR] Data preparation failed:")
        print(result.stderr)
        raise RuntimeError(f"Data preparation failed for {config_name}")
    
    # Locate the generated parquet file (recursively search in the output directory)
    # The data preparation script should generate exactly one parquet file
    parquet_files = list(config_output_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet file generated in {config_output_dir}")
    
    # Use the first (and typically only) parquet file found
    parquet_file = parquet_files[0]
    print(f"[OK] Data prepared: {parquet_file}")
    return str(parquet_file)


def index_data(
    parquet_file: str,
    index_name: str,
    api_keys_path: str,
) -> None:
    """
    Index the prepared parquet data into a Pinecone vector database.
    
    This function calls the indexing.py script to create or update a Pinecone index
    with the prepared document chunks. Each configuration gets its own Pinecone index
    to allow for independent evaluation.
    
    Args:
        parquet_file: Path to the parquet file containing prepared document chunks
        index_name: Name of the Pinecone index to create/update (must be unique per config)
        api_keys_path: Path to JSON file containing API keys (Pinecone, OpenAI, etc.)
    
    Raises:
        RuntimeError: If the indexing subprocess fails or returns a non-zero exit code
    
    Note:
        The index name should be unique for each configuration to avoid conflicts.
        Pinecone index names must follow specific naming conventions (lowercase, alphanumeric, hyphens).
    """
    print(f"\n[STEP] Indexing data to Pinecone index: {index_name}...")
    
    # Build command to run the indexing script as a subprocess
    # This processes the parquet file and uploads embeddings to Pinecone
    cmd = [
        sys.executable,
        "indexing.py",
        "--prepared_file", parquet_file,
        "--api_keys_path", api_keys_path,
        "--index_name", index_name,
    ]
    
    # Execute indexing with real-time output streaming to show progress bars
    # Write directly to terminal (stdout/stderr) so tqdm progress bars display properly
    # Capture stderr separately for error reporting if needed
    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,  # Write directly to terminal for real-time progress bars
        stderr=subprocess.PIPE,  # Capture stderr for error reporting
        text=True
    )
    
    # Wait for process to complete and capture stderr
    _, stderr_output = process.communicate()
    returncode = process.returncode
    
    # Verify indexing completed successfully
    if returncode != 0:
        print(f"[ERROR] Indexing failed:")
        if stderr_output:
            print(stderr_output)
        raise RuntimeError(f"Indexing failed for {index_name}")
    
    print(f"[OK] Data indexed to {index_name}")


def run_evaluation_for_config(
    config_name: str,
    index_name: str,
    output_dir: str,
    queries: List[Dict],
    api_keys_path: str,
    strategies: List[str] = None,
    top_k: int = DEFAULT_EVALUATION_TOP_K,
) -> str:
    """
    Run retrieval evaluation for a specific chunk configuration.
    
    This function evaluates the retrieval performance of a specific chunk configuration
    by running queries against the corresponding Pinecone index. It tests multiple
    retrieval strategies (baseline, sentence-based, adaptive) and generates detailed
    evaluation metrics.
    
    Args:
        config_name: Name identifier for this configuration (used for output directory)
        index_name: Name of the Pinecone index to query (should match the indexed data)
        output_dir: Base directory where evaluation results will be stored
        queries: List of query dictionaries, each containing query text and ground truth
        api_keys_path: Path to JSON file containing API keys for services
        strategies: List of retrieval strategies to evaluate. Defaults to ["baseline", "sentence", "adaptive"]
        top_k: Number of top chunks to retrieve per query (default: 5)
    
    Returns:
        str: Path to the evaluation results directory containing CSV files and metrics
    
    Note:
        Evaluation results are saved in: output_dir/config_name/evaluation_results/
        The results include strategy_statistics.csv with performance metrics for each strategy.
    """
    # Set default strategies if none provided
    # These represent different chunk retrieval approaches:
    # - baseline: Standard vector similarity search
    # - sentence: Sentence-level chunking and retrieval
    # - adaptive: Adaptive chunking based on content structure
    if strategies is None:
        strategies = DEFAULT_EVALUATION_STRATEGIES
    
    # Create output directory for this configuration's evaluation results
    eval_output_dir = Path(output_dir) / config_name / "evaluation_results"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[STEP] Running evaluation for {config_name}...")
    print(f"       Using index: {index_name}")
    
    # Run the evaluation using the imported evaluation function
    # This will execute all queries against the specified index and generate metrics
    run_evaluation(
        queries=queries,
        output_dir=str(eval_output_dir),
        strategies=strategies,
        top_k=top_k,
        api_keys_path=api_keys_path,
        include_baselines=False,  # Skip baseline strategies as we're testing chunk configs
        index_name=index_name,
    )
    
    print(f"[OK] Evaluation complete: {eval_output_dir}")
    return str(eval_output_dir)


def load_configs(config_file: str) -> List[Dict]:
    """
    Load and validate chunk configurations from a JSON file.
    
    This function reads a JSON file containing an array of configuration objects,
    each specifying chunk size and overlap parameters to test. It validates that
    all required fields are present in each configuration.
    
    Args:
        config_file: Path to JSON file containing configuration array
    
    Returns:
        List[Dict]: List of configuration dictionaries, each containing:
            - name: Unique identifier for the configuration
            - chunk_chars: Maximum characters per chunk
            - chunk_overlap: Character overlap between consecutive chunks
    
    Raises:
        FileNotFoundError: If the config file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        ValueError: If any configuration is missing required fields
    
    Example:
        Config file should contain:
        [
            {"name": "small", "chunk_chars": 500, "chunk_overlap": 100},
            {"name": "large", "chunk_chars": 2000, "chunk_overlap": 400}
        ]
    """
    with open(config_file, "r", encoding="utf-8") as f:
        configs = json.load(f)
    
    # Validate that each configuration has all required fields
    # This prevents runtime errors later when processing configurations
    for config in configs:
        required = ["name", "chunk_chars", "chunk_overlap"]
        for field in required:
            if field not in config:
                raise ValueError(f"Config missing required field: {field}")
    
    return configs


def compare_config_results(results_dir: str) -> None:
    """
    Compare evaluation results across all tested configurations.
    
    This function aggregates statistics from all configuration evaluations and creates
    a unified comparison CSV file. It reads the strategy_statistics.csv file from each
    configuration's evaluation results and combines them with configuration identifiers.
    
    The comparison helps identify which chunk size and overlap combinations perform best
    across different retrieval strategies.
    
    Args:
        results_dir: Base directory containing subdirectories for each configuration
    
    Note:
        This function requires pandas. It will gracefully skip comparison if:
        - No evaluation results are found
        - CSV files are missing or malformed
        - pandas is not available
    
    Output:
        Creates config_comparison.csv in the results_dir with aggregated statistics.
        Displays a summary table showing average scores grouped by configuration and strategy.
    """
    results_path = Path(results_dir)
    # Find all subdirectories (each represents a tested configuration)
    config_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    
    # Collect statistics from each configuration's evaluation results
    all_stats = []
    for config_dir in config_dirs:
        # Each configuration should have evaluation_results/strategy_statistics.csv
        stats_file = config_dir / "evaluation_results" / "strategy_statistics.csv"
        if stats_file.exists():
            import pandas as pd
            # Read the statistics CSV and add configuration name as a column
            df = pd.read_csv(stats_file)
            df["config_name"] = config_dir.name
            all_stats.append(df)
    
    # Combine all statistics into a single DataFrame for comparison
    if all_stats:
        import pandas as pd
        combined = pd.concat(all_stats, ignore_index=True)
        comparison_file = results_path / "config_comparison.csv"
        combined.to_csv(comparison_file, index=False, encoding="utf-8")
        print(f"\n[OK] Configuration comparison saved to: {comparison_file}")
        
        # Display summary statistics grouped by configuration and strategy
        # This shows average performance metrics for easy comparison
        print("\nSummary:")
        print(combined.groupby(["config_name", "strategy"])[["avg_score_mean", "namespace_correct_mean"]].mean())


def main():
    """
    Main entry point for the chunk configuration evaluation script.
    
    This function orchestrates the complete evaluation pipeline:
    1. Parses command-line arguments
    2. Loads chunk configurations from JSON file
    3. Loads test queries (from testset or uses defaults)
    4. For each configuration:
       a. Prepares data with specific chunk parameters
       b. Indexes data into Pinecone
       c. Runs retrieval evaluation
    5. Generates comparison report across all configurations
    
    The script supports skipping preparation and indexing steps for faster iteration
    when re-running evaluations with existing data.
    """
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
        default="evaluation/chunks_config_comparison_eval_results",
        help="Base directory for all configuration outputs",
    )
    parser.add_argument(
        "--api_keys_path",
        type=str,
        default=DEFAULT_API_KEYS_PATH,
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
        default=DEFAULT_EVALUATION_STRATEGIES,
        help="Chunking strategies to evaluate",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_EVALUATION_TOP_K,
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
    
    # Load chunk configurations from the specified JSON file
    # Each configuration defines chunk_chars and chunk_overlap to test
    configs = load_configs(args.configs)
    print(f"\n[INFO] Loaded {len(configs)} configurations to test")
    
    # Load test queries for evaluation
    # Priority: 1) User-specified testset, 2) Default testset, 3) Hardcoded default queries
    if args.testset_file and os.path.exists(args.testset_file):
        queries = load_testset_with_ground_truth(args.testset_file)
        print(f"[INFO] Using testset with ground truth: {args.testset_file}")
    else:
        # Try default testset location (relative to project root)
        default_testset = Path("tests/embedding_testset.json")
        if default_testset.exists():
            queries = load_testset_with_ground_truth(str(default_testset))
            print(f"[INFO] Using default testset: {default_testset}")
        else:
            # Fall back to hardcoded default queries if no testset file is found
            queries = DEFAULT_QUERIES
            print(f"[INFO] Using default queries")
    
    # Create output directory structure for all configuration results
    output_base = Path(args.output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Process each configuration sequentially
    # Each configuration goes through: preparation -> indexing -> evaluation
    for config in configs:
        config_name = config["name"]
        chunk_chars = config["chunk_chars"]
        chunk_overlap = config["chunk_overlap"]
        
        # Sanitize config_name for Pinecone index naming requirements
        # Pinecone index names must contain only lowercase alphanumeric characters and hyphens
        # Replace underscores with hyphens to ensure compatibility
        safe_config_name = config_name.replace("_", "-")
        index_name = f"haifa-municipality-rag-{safe_config_name}"
        
        try:
            # Step 1: Prepare data with the specified chunk configuration
            # This processes the input JSON and creates chunks according to chunk_chars and chunk_overlap
            if not args.skip_preparation:
                parquet_file = prepare_data_for_config(
                    input_json=args.input_json,
                    output_dir=str(output_base),
                    chunk_chars=chunk_chars,
                    chunk_overlap=chunk_overlap,
                    config_name=config_name,
                )
            else:
                # Skip preparation step - locate existing prepared parquet file
                # Useful when re-running evaluations without regenerating data
                # Only needed if we're going to index (not needed if skip_indexing is also set)
                if not args.skip_indexing:
                    config_dir = output_base / config_name
                    parquet_files = list(config_dir.glob("**/*.parquet"))
                    if not parquet_files:
                        raise FileNotFoundError(f"No parquet file found for {config_name}")
                    parquet_file = str(parquet_files[0])
                    print(f"[INFO] Using existing prepared file: {parquet_file}")
                else:
                    # Both skip_preparation and skip_indexing are set
                    # Parquet file not needed since we're only running evaluation
                    parquet_file = None
                    print(f"[INFO] Skipping preparation and indexing - parquet file not required")
            
            # Step 2: Index the prepared data into Pinecone
            # Creates a vector index that can be queried for retrieval evaluation
            if not args.skip_indexing:
                index_data(
                    parquet_file=parquet_file,
                    index_name=index_name,
                    api_keys_path=args.api_keys_path,
                )
            else:
                # Skip indexing step - assume index already exists
                # Useful when re-running evaluations with existing indexes
                print(f"[INFO] Skipping indexing, using existing index: {index_name}")
            
            # Step 3: Run retrieval evaluation against the indexed data
            # Tests multiple retrieval strategies and generates performance metrics
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
            # Handle errors gracefully - log the error but continue with other configurations
            # This allows partial results even if one configuration fails
            print(f"\n[ERROR] Failed to process configuration {config_name}: {e}")
            import traceback
            traceback.print_exc()
            continue  # Continue processing remaining configurations
    
    # Final step: Compare results across all tested configurations
    # Generates a unified comparison CSV and summary statistics
    print(f"\n{'='*70}")
    print("COMPARING CONFIGURATIONS")
    print(f"{'='*70}")
    try:
        compare_config_results(str(output_base))
    except Exception as e:
        # Comparison is non-critical - if it fails, individual results are still available
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


