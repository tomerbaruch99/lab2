"""
Analysis Script for Evaluation Results
======================================

This script analyzes evaluation results from locally stored CSV files.
It does NOT query Pinecone, Gemini, or any external APIs.
All analysis is based on pre-generated CSV files from generate_evaluation_results.py

Usage:
    python evaluation/analyze_results.py \
        --results_dir evaluation/evaluation_results
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script mode
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.quantitative_report_generator import generate_quantitative_report
from evaluation.visualization_utils import create_all_basic_visualizations


def load_results(results_dir: Path):
    """Load all result files from the results directory."""
    results_dir = Path(results_dir)
    
    # Required file
    results_csv = results_dir / "evaluation_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Results file not found: {results_csv}")
    
    df = pd.read_csv(results_csv, encoding="utf-8")
    
    # Optional files
    strategy_stats = None
    namespace_stats = None
    score_distribution = None
    improvements = None
    significance_tests = None
    
    stats_csv = results_dir / "strategy_statistics.csv"
    if stats_csv.exists():
        strategy_stats = pd.read_csv(stats_csv, encoding="utf-8")
    
    namespace_csv = results_dir / "namespace_statistics.csv"
    if namespace_csv.exists():
        namespace_stats = pd.read_csv(namespace_csv, encoding="utf-8")
    
    score_dist_csv = results_dir / "score_distribution.csv"
    if score_dist_csv.exists():
        score_distribution = pd.read_csv(score_dist_csv, encoding="utf-8")
    
    improvements_csv = results_dir / "improvements_over_baselines.csv"
    if improvements_csv.exists():
        improvements = pd.read_csv(improvements_csv, encoding="utf-8")
    
    sig_tests_csv = results_dir / "statistical_significance_tests.csv"
    if sig_tests_csv.exists():
        significance_tests = pd.read_csv(sig_tests_csv, encoding="utf-8")
    
    return {
        "results": df,
        "strategy_stats": strategy_stats,
        "namespace_stats": namespace_stats,
        "score_distribution": score_distribution,
        "improvements": improvements,
        "significance_tests": significance_tests,
    }


def generate_visualizations(data: dict, output_dir: Path):
    """Generate basic visualization plots focusing on distributions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        create_all_basic_visualizations(data, output_dir)
    except Exception as e:
        print(f"[WARN] Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Also create namespace heatmap if available
    namespace_stats = data.get("namespace_stats")
    if namespace_stats is not None and len(namespace_stats) > 0:
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            namespace_stats_df = pd.DataFrame(namespace_stats) if isinstance(namespace_stats, list) else namespace_stats
            
            pivot = namespace_stats_df.pivot_table(
                values='accuracy',
                index='expected_namespace',
                columns='strategy',
                fill_value=0
            )
            
            sns.heatmap(pivot, annot=True, fmt='.2%', cmap='YlOrRd', 
                       cbar_kws={'label': 'Accuracy'}, ax=ax)
            ax.set_title('Namespace Detection Accuracy Heatmap', fontsize=14, fontweight='bold')
            ax.set_xlabel('Strategy')
            ax.set_ylabel('Expected Namespace')
            
            plt.tight_layout()
            plt.savefig(output_dir / '8_namespace_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"[OK] Saved namespace accuracy heatmap to {output_dir / '8_namespace_accuracy_heatmap.png'}")
            plt.close()
        except Exception as e:
            print(f"[WARN] Error creating namespace heatmap: {e}")


def print_summary(data: dict):
    """Print summary statistics to console."""
    df = data["results"]
    strategy_stats = data["strategy_stats"]
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal Queries: {df['query'].nunique()}")
    print(f"Strategies Evaluated: {', '.join(df['strategy'].unique())}")
    print(f"Total Results: {len(df)}")
    
    if strategy_stats is not None:
        print("\nStrategy Performance:")
        print("-" * 70)
        for _, row in strategy_stats.iterrows():
            print(f"\n{row['strategy'].upper()}:")
            print(f"  Average Score: {row['avg_score_mean']:.4f} (±{row.get('avg_score_std', 0):.4f})")
            if 'namespace_correct_mean' in row:
                print(f"  Namespace Accuracy: {row['namespace_correct_mean']:.2%}")
    
    improvements = data["improvements"]
    if improvements is not None and len(improvements) > 0:
        print("\n\nBaseline Comparisons:")
        print("-" * 70)
        for _, row in improvements.iterrows():
            print(f"\n{row['main_strategy']} vs {row['baseline_strategy']}:")
            print(f"  Mean Improvement: +{row['mean_improvement_pct']:.2f}%")
            print(f"  Queries Better: {row['queries_better']}/{row['total_queries']} ({row['proportion_better_pct']:.1f}%)")
    
    significance_tests = data["significance_tests"]
    if significance_tests is not None and len(significance_tests) > 0:
        print("\n\nStatistical Significance:")
        print("-" * 70)
        for _, row in significance_tests.iterrows():
            if 'error' not in str(row.get('error', '')):
                sig = "✓ Significant" if row.get('significant', False) else "✗ Not Significant"
                print(f"{row['strategy1']} vs {row['strategy2']}: {sig} (p={row.get('p_value', 'N/A')})")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze evaluation results from local CSV files (no API access required)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="evaluation/evaluation_results",
        help="Directory containing evaluation result CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save analysis outputs (default: same as results_dir)",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip generating visualization plots",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Skip printing summary statistics",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS ANALYSIS")
    print("=" * 70)
    print(f"\n📂 Loading results from: {results_dir}")
    print(f"📊 Analysis outputs to: {output_dir}")
    print("\n[INFO] This script does NOT access Pinecone, Gemini, or any external APIs")
    print("       All analysis is based on locally stored CSV files.\n")
    
    # Load data
    try:
        data = load_results(results_dir)
        print(f"✅ Loaded {len(data['results'])} evaluation results")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease run generate_evaluation_results.py first to generate results.")
        return 1
    
    # Generate visualizations
    if not args.no_plots:
        print("\n[STEP] Generating visualizations...")
        try:
            generate_visualizations(data, output_dir)
        except Exception as e:
            print(f"[WARN] Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if not args.no_summary:
        print_summary(data)
    
    # Regenerate quantitative report if data is available
    print("\n[STEP] Generating quantitative analysis report...")
    try:
        from evaluation.generate_evaluation_results import (
            compute_comparative_statistics,
            compute_score_distribution,
        )
        
        # Recompute statistics if needed
        comparative_stats = compute_comparative_statistics(data['results'])
        score_distribution = compute_score_distribution(data['results'])
        
        strategy_stats = data['strategy_stats']
        if strategy_stats is None:
            from evaluation.generate_evaluation_results import compute_strategy_statistics
            strategy_stats = compute_strategy_statistics(data['results'])
        
        report_path = output_dir / "quantitative_analysis_report.md"
        generate_quantitative_report(
            comparative_stats,
            strategy_stats,
            score_distribution,
            output_path=report_path,
        )
        print(f"[OK] Quantitative analysis report saved to {report_path}")
    except Exception as e:
        print(f"[WARN] Could not generate quantitative report: {e}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

