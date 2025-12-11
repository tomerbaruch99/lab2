"""
Visualize LLM Judge Evaluation Results
======================================

This script creates comprehensive visualizations for LLM judge evaluation results,
including enrichment/reranking comparisons.

Usage:
    python evaluation/visualize_llm_judge_results.py \
        --results_dir evaluation/llm_judge_results
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Simple visualization functions for LLM judge results


def load_llm_judge_results(results_dir: Path):
    """Load LLM judge results from CSV files."""
    results_dir = Path(results_dir)
    
    results_csv = results_dir / "llm_judge_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"LLM judge results file not found: {results_csv}")
    
    df = pd.read_csv(results_csv, encoding="utf-8")
    
    comparison_csv = results_dir / "enrichment_reranking_comparison.csv"
    comparison_df = None
    if comparison_csv.exists():
        comparison_df = pd.read_csv(comparison_csv, encoding="utf-8")
    
    stats_csv = results_dir / "llm_judge_statistics.csv"
    stats_df = None
    if stats_csv.exists():
        stats_df = pd.read_csv(stats_csv, encoding="utf-8")
    
    return {
        "results": df,
        "comparison": comparison_df,
        "statistics": stats_df,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LLM judge evaluation results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="evaluation/llm_judge_results",
        help="Directory containing LLM judge result CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualizations (default: same as results_dir)",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    
    print("\n" + "=" * 70)
    print("LLM JUDGE RESULTS VISUALIZATION")
    print("=" * 70)
    print(f"\n📂 Loading results from: {results_dir}")
    print(f"📊 Saving visualizations to: {output_dir}\n")
    
    # Load data
    try:
        data = load_llm_judge_results(results_dir)
        print(f"✅ Loaded {len(data['results'])} LLM judge results")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease run run_llm_judge_evaluation.py first to generate results.")
        return 1
    
    # Check if enrichment/reranking data is available
    df = data["results"]
    has_enrichment = 'query_enrichment' in df.columns
    has_reranking = 'reranking' in df.columns
    
    print("\n[STEP] Creating LLM judge visualizations...")
    
    # Determine which score column to use
    score_col = 'overall_score' if 'overall_score' in df.columns else None
    if score_col is None:
        metrics = ['correctness', 'faithfulness', 'completeness', 'conciseness']
        available = [m for m in metrics if m in df.columns]
        if available:
            score_col = available[0]
    
    if score_col is None:
        print("⚠️ No score columns found in results")
        return 0
    
    try:
        # 1. Bar chart - Average scores by configuration
        fig, ax = plt.subplots(figsize=(12, 7))
        
        if has_enrichment and has_reranking:
            # Group by enrichment/reranking configuration
            df['config'] = df.apply(
                lambda row: f"Enrich: {row['query_enrichment']}, Rerank: {row['reranking']}", axis=1
            )
            group_col = 'config'
        else:
            group_col = 'strategy'
        
        config_scores = df.groupby(group_col)[score_col].agg(['mean', 'std']).sort_values('mean', ascending=False)
        bars = ax.barh(range(len(config_scores)), config_scores['mean'],
                      xerr=config_scores['std'], capsize=5,
                      color=sns.color_palette("husl", len(config_scores)), 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(len(config_scores)))
        ax.set_yticklabels(config_scores.index, fontsize=10)
        ax.set_xlabel(f'{score_col.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_title(f'Average {score_col.replace("_", " ").title()} by Configuration', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
        
        # Add value labels
        for i, (bar, (mean, std)) in enumerate(zip(bars, config_scores.itertuples(index=False))):
            ax.text(mean + std + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{mean:.3f}±{std:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / '1_llm_judge_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved bar chart to {output_dir / '1_llm_judge_bar.png'}")
        
        # 2. Box plot - Score distributions
        fig, ax = plt.subplots(figsize=(12, 7))
        configs = df[group_col].unique()
        data_for_box = [df[df[group_col] == c][score_col].values for c in configs]
        
        bp = ax.boxplot(data_for_box, labels=configs, patch_artist=True,
                       showmeans=True, meanline=True, showfliers=True)
        
        colors = sns.color_palette("husl", len(configs))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        
        ax.set_ylabel(f'{score_col.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_title(f'{score_col.replace("_", " ").title()} Distribution (Box Plot)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / '2_llm_judge_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved box plot to {output_dir / '2_llm_judge_boxplot.png'}")
        
        # 3. Histogram - Score distributions
        n_configs = len(configs)
        fig, axes = plt.subplots(1, min(n_configs, 4), figsize=(5*min(n_configs, 4), 5))
        if n_configs == 1:
            axes = [axes]
        elif n_configs > 4:
            axes = axes[:4]  # Limit to 4 subplots
        
        for idx, (config, color) in enumerate(zip(configs[:4], colors[:4])):
            ax = axes[idx] if n_configs > 1 else axes[0]
            scores = df[df[group_col] == config][score_col].values
            
            ax.hist(scores, bins=15, alpha=0.7, color=color, edgecolor='black', linewidth=1.5)
            ax.axvline(scores.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {scores.mean():.3f}')
            ax.axvline(np.median(scores), color='blue', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(scores):.3f}')
            
            ax.set_xlabel(f'{score_col.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'{config}\n(n={len(scores)})', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_xlim(0, 1)
        
        plt.suptitle(f'{score_col.replace("_", " ").title()} Distribution Histogram', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / '3_llm_judge_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved histogram to {output_dir / '3_llm_judge_histogram.png'}")
        
        # 4. Scatter plot - Individual scores
        fig, ax = plt.subplots(figsize=(12, 7))
        x_positions = {config: idx for idx, config in enumerate(configs)}
        
        for config, color in zip(configs, colors):
            config_data = df[df[group_col] == config]
            scores = config_data[score_col].values
            x_jitter = np.random.normal(x_positions[config], 0.1, len(scores))
            
            ax.scatter(x_jitter, scores, alpha=0.6, color=color, s=50,
                      edgecolors='black', linewidth=0.5, label=config)
            mean_score = scores.mean()
            ax.axhline(mean_score, color=color, linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(f'{score_col.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_title(f'Individual Query Scores by Configuration', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(output_dir / '4_llm_judge_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved scatter plot to {output_dir / '4_llm_judge_scatter.png'}")
        
    except Exception as e:
        print(f"[WARN] Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    if data.get("comparison") is not None:
        print("\n" + "=" * 70)
        print("ENRICHMENT/RERANKING COMPARISON SUMMARY")
        print("=" * 70)
        print(data["comparison"].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

