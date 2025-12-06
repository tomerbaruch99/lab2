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
    """Generate visualization plots from loaded data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = data["results"]
    strategy_stats = data["strategy_stats"]
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # 1. Strategy Comparison
    if strategy_stats is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Average scores
        ax = axes[0, 0]
        strategies = strategy_stats['strategy']
        means = strategy_stats['avg_score_mean']
        stds = strategy_stats.get('avg_score_std', [0] * len(strategies))
        
        bars = ax.bar(strategies, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_title('Average Retrieval Scores by Strategy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Score')
        ax.set_xlabel('Strategy')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                   f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Namespace accuracy
        ax = axes[0, 1]
        if 'namespace_correct_mean' in strategy_stats.columns:
            accuracies = strategy_stats['namespace_correct_mean']
            bars = ax.bar(strategies, accuracies, alpha=0.7, color='coral')
            ax.set_title('Namespace Detection Accuracy', fontsize=14, fontweight='bold')
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Strategy')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.2%}',
                       ha='center', va='bottom', fontsize=10)
        
        # Score distribution (box plot)
        ax = axes[1, 0]
        strategy_list = df['strategy'].unique()
        score_data = [df[df['strategy'] == s]['avg_score'].values for s in strategy_list]
        bp = ax.boxplot(score_data, labels=strategy_list, patch_artist=True)
        ax.set_title('Score Distribution by Strategy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Score')
        ax.set_xlabel('Strategy')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Color the boxes
        colors = plt.cm.Set3(range(len(strategy_list)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Query category performance
        ax = axes[1, 1]
        if 'category' in df.columns:
            category_perf = df.groupby(['strategy', 'category'])['avg_score'].mean().unstack(fill_value=0)
            category_perf.plot(kind='bar', ax=ax, alpha=0.7)
            ax.set_title('Performance by Query Category', fontsize=14, fontweight='bold')
            ax.set_ylabel('Average Score')
            ax.set_xlabel('Strategy')
            ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'strategy_comparison.png', dpi=300, bbox_inches='tight')
        print(f"[OK] Saved strategy comparison plot to {output_dir / 'strategy_comparison.png'}")
        plt.close()
    
    # 2. Namespace Accuracy Heatmap
    namespace_stats = data["namespace_stats"]
    if namespace_stats is not None and len(namespace_stats) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Pivot for heatmap
        pivot = namespace_stats.pivot_table(
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
        plt.savefig(output_dir / 'namespace_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"[OK] Saved namespace accuracy heatmap to {output_dir / 'namespace_accuracy_heatmap.png'}")
        plt.close()
    
    # 3. Score Distribution
    score_dist = data["score_distribution"]
    if score_dist is not None and len(score_dist) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        strategies = score_dist['strategy'].unique()
        categories = ['Excellent (≥0.8)', 'Good (0.6-0.8)', 'Moderate (0.4-0.6)', 'Poor (<0.4)']
        
        # Prepare data for stacked bar chart
        category_order = ['Excellent (≥0.8)', 'Good (0.6-0.8)', 'Moderate (0.4-0.6)', 'Poor (<0.4)']
        bottom = np.zeros(len(strategies))
        
        colors_map = {
            'Excellent (≥0.8)': '#2ecc71',
            'Good (0.6-0.8)': '#3498db',
            'Moderate (0.4-0.6)': '#f39c12',
            'Poor (<0.4)': '#e74c3c'
        }
        
        for cat in category_order:
            values = []
            for strat in strategies:
                cat_data = score_dist[(score_dist['strategy'] == strat) & 
                                     (score_dist['score_category'] == cat)]
                if len(cat_data) > 0:
                    values.append(cat_data.iloc[0]['proportion'])
                else:
                    values.append(0)
            
            ax.bar(strategies, values, bottom=bottom, label=cat, 
                  color=colors_map.get(cat, '#95a5a6'), alpha=0.8)
            bottom += np.array(values)
        
        ax.set_title('Score Distribution by Strategy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Proportion of Queries (%)')
        ax.set_xlabel('Strategy')
        ax.set_ylim([0, 100])
        ax.legend(title='Score Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
        print(f"[OK] Saved score distribution plot to {output_dir / 'score_distribution.png'}")
        plt.close()


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

