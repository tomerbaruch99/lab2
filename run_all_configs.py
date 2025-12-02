"""
Helper script to prepare and index multiple chunk configurations for comparison.

This script:
1. Runs data preparation for multiple chunk size/overlap configurations
2. Optionally indexes each configuration to separate Pinecone indexes
3. Generates a comparison report

Usage:
    python run_all_configs.py --prepare-only
    python run_all_configs.py --index-all --index-prefix haifa-config
"""

import argparse
import subprocess
import sys
from pathlib import Path

from utils import DEFAULT_API_KEYS_PATH


# Default configurations to test
DEFAULT_CONFIGS = [
    (500, 100),   # Small chunks, small overlap
    (750, 150),   # Medium-small chunks
    (1000, 200),  # Default
    (1500, 300),  # Larger chunks
    (2000, 400),  # Large chunks
]


def run_preparation(input_json: str, out_dir: str, configs: list):
    """Run data preparation for all configurations."""
    print("="*60)
    print("PREPARING DATA FOR ALL CONFIGURATIONS")
    print("="*60)
    
    for chunk_chars, chunk_overlap in configs:
        config_suffix = f"chunk{chunk_chars}_overlap{chunk_overlap}"
        print(f"\n[CONFIG] {config_suffix}")
        print("-" * 60)
        
        cmd = [
            sys.executable,
            "scrape_and_prepare_data/data_preparation.py",
            "--input_json", input_json,
            "--out_dir", out_dir,
            "--chunk_chars", str(chunk_chars),
            "--chunk_overlap", str(chunk_overlap),
            "--config_suffix", config_suffix,
        ]
        
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"[ERROR] Failed to prepare data for {config_suffix}")
            return False
    
    print("\n" + "="*60)
    print("[OK] All configurations prepared successfully!")
    print("="*60)
    return True


def run_indexing(prepared_dir: str, configs: list, index_prefix: str, api_keys_path: str, namespace: str = None):
    """Index all configurations to separate Pinecone indexes."""
    print("\n" + "="*60)
    print("INDEXING ALL CONFIGURATIONS")
    print("="*60)
    if namespace:
        print(f"[INFO] Using namespace: {namespace}")
    
    for chunk_chars, chunk_overlap in configs:
        config_suffix = f"chunk{chunk_chars}_overlap{chunk_overlap}"
        index_name = f"{index_prefix}-{config_suffix}"
        
        print(f"\n[CONFIG] {config_suffix} -> Index: {index_name}")
        if namespace:
            print(f"         Namespace: {namespace}")
        print("-" * 60)
        
        cmd = [
            sys.executable,
            "indexing.py",
            "--prepared_dir", prepared_dir,
            "--config", config_suffix,
            "--index_name", index_name,
            "--api_keys_path", api_keys_path,
        ]
        if namespace:
            cmd.extend(["--namespace", namespace])
        
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"[ERROR] Failed to index {config_suffix}")
            return False
    
    print("\n" + "="*60)
    print("[OK] All configurations indexed successfully!")
    print("="*60)
    return True


def generate_comparison_report(prepared_dir: str, configs: list):
    """Generate a comparison report of all configurations."""
    import pandas as pd
    
    print("\n" + "="*60)
    print("GENERATING COMPARISON REPORT")
    print("="*60)
    
    report_data = []
    for chunk_chars, chunk_overlap in configs:
        config_suffix = f"chunk{chunk_chars}_overlap{chunk_overlap}"
        parquet_path = Path(prepared_dir) / f"haifa_paragraph_index_config_{config_suffix}.parquet"
        
        if not parquet_path.exists():
            print(f"[WARN] File not found: {parquet_path}")
            continue
        
        try:
            df = pd.read_parquet(parquet_path)
            report_data.append({
                "config": config_suffix,
                "chunk_chars": chunk_chars,
                "chunk_overlap": chunk_overlap,
                "total_chunks": len(df),
                "avg_chunk_length": df["text"].str.len().mean(),
                "min_chunk_length": df["text"].str.len().min(),
                "max_chunk_length": df["text"].str.len().max(),
                "unique_docs": df["doc_id"].nunique(),
                "avg_chunks_per_doc": len(df) / df["doc_id"].nunique(),
            })
        except Exception as e:
            print(f"[ERROR] Failed to read {parquet_path}: {e}")
    
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values("chunk_chars")
        
        print("\nConfiguration Comparison:")
        print("="*60)
        print(report_df.to_string(index=False))
        print("="*60)
        
        # Save report
        report_path = Path(prepared_dir) / "config_comparison_report.csv"
        report_df.to_csv(report_path, index=False)
        print(f"\n[OK] Report saved to: {report_path}")
    else:
        print("[WARN] No data to compare")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and index multiple chunk configurations for comparison"
    )
    parser.add_argument("--input_json", type=str,
                        default="./scrape_and_prepare_data/haifa_scraped.json",
                        help="Path to input JSON file")
    parser.add_argument("--out_dir", type=str,
                        default="./scrape_and_prepare_data/haifa_prepared_data",
                        help="Output directory for prepared data")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only run data preparation, skip indexing")
    parser.add_argument("--index-all", action="store_true",
                        help="Index all configurations to separate Pinecone indexes")
    parser.add_argument("--index-prefix", type=str, default="haifa-config",
                        help="Prefix for Pinecone index names")
    parser.add_argument("--api_keys_path", type=str, default=DEFAULT_API_KEYS_PATH,
                        help="Path to API keys file")
    parser.add_argument("--namespace", type=str, default=None,
                        help="Optional namespace for dev/prod/language separation (e.g., 'dev', 'prod', 'hebrew', 'arabic')")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated configs in format 'chunk:overlap,chunk:overlap' (e.g., '500:100,1000:200')")
    
    args = parser.parse_args()
    
    # Parse configurations
    if args.configs:
        configs = []
        for config_str in args.configs.split(","):
            chunk_chars, chunk_overlap = map(int, config_str.split(":"))
            configs.append((chunk_chars, chunk_overlap))
    else:
        configs = DEFAULT_CONFIGS
    
    print(f"Configurations to process: {len(configs)}")
    for chunk_chars, chunk_overlap in configs:
        print(f"  - chunk_chars={chunk_chars}, chunk_overlap={chunk_overlap}")
    
    # Run preparation
    if not run_preparation(args.input_json, args.out_dir, configs):
        sys.exit(1)
    
    # Generate comparison report
    generate_comparison_report(args.out_dir, configs)
    
    # Run indexing if requested
    if args.index_all and not args.prepare_only:
        if not run_indexing(args.out_dir, configs, args.index_prefix, args.api_keys_path, args.namespace):
            sys.exit(1)


if __name__ == "__main__":
    main()

