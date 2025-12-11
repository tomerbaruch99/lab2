"""
Basic Visualization Utilities for Evaluation Results
====================================================

This module provides clear, explanatory visualization functions focusing on distributions:
- Bar charts for comparing means
- Box plots for score distributions
- Histograms for score distributions
- Scatter plots for relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def create_strategy_comparison_bar_chart(
    strategy_stats: pd.DataFrame,
    output_path: Path,
    title: str = "Strategy Comparison - Average Scores"
) -> None:
    """Create a simple bar chart comparing average scores across strategies."""
    if strategy_stats is None or len(strategy_stats) == 0:
        print("⚠️ No strategy statistics available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = strategy_stats['strategy'].values
    means = strategy_stats['avg_score_mean'].values
    stds = strategy_stats.get('avg_score_std', pd.Series([0] * len(strategies))).values
    
    bars = ax.bar(strategies, means, yerr=stds, capsize=5, alpha=0.8, 
                  color=sns.color_palette("husl", len(strategies)), edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Average Retrieval Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
               f'{mean:.3f}\n±{std:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved bar chart to {output_path}")


def create_score_distribution_boxplot(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Score Distribution by Strategy"
) -> None:
    """Create a box plot showing score distributions for each strategy."""
    if 'strategy' not in df.columns or 'avg_score' not in df.columns:
        print("⚠️ Required columns (strategy, avg_score) not found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    strategies = df['strategy'].unique()
    data_for_box = [df[df['strategy'] == s]['avg_score'].values for s in strategies]
    
    bp = ax.boxplot(data_for_box, labels=strategies, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=True)
    
    # Color the boxes
    colors = sns.color_palette("husl", len(strategies))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Style the other elements
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    
    ax.set_ylabel('Retrieval Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved box plot to {output_path}")


def create_score_histogram(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Score Distribution Histogram"
) -> None:
    """Create histograms showing score distributions for each strategy."""
    if 'strategy' not in df.columns or 'avg_score' not in df.columns:
        print("⚠️ Required columns (strategy, avg_score) not found")
        return
    
    strategies = df['strategy'].unique()
    n_strategies = len(strategies)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_strategies, figsize=(5*n_strategies, 5))
    if n_strategies == 1:
        axes = [axes]
    
    colors = sns.color_palette("husl", n_strategies)
    
    for idx, (strategy, color) in enumerate(zip(strategies, colors)):
        ax = axes[idx]
        scores = df[df['strategy'] == strategy]['avg_score'].values
        
        ax.hist(scores, bins=15, alpha=0.7, color=color, edgecolor='black', linewidth=1.5)
        ax.axvline(scores.mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {scores.mean():.3f}')
        ax.axvline(np.median(scores), color='blue', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(scores):.3f}')
        
        ax.set_xlabel('Retrieval Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{strategy}\n(n={len(scores)})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved histogram to {output_path}")


def create_score_scatter_plot(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Score Scatter Plot by Strategy"
) -> None:
    """Create a scatter plot showing individual query scores by strategy."""
    if 'strategy' not in df.columns or 'avg_score' not in df.columns:
        print("⚠️ Required columns (strategy, avg_score) not found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    strategies = df['strategy'].unique()
    colors = sns.color_palette("husl", len(strategies))
    
    # Create jitter for x-axis
    x_positions = {strat: idx for idx, strat in enumerate(strategies)}
    
    for strategy, color in zip(strategies, colors):
        strategy_data = df[df['strategy'] == strategy]
        scores = strategy_data['avg_score'].values
        
        # Add jitter
        x_jitter = np.random.normal(x_positions[strategy], 0.1, len(scores))
        
        ax.scatter(x_jitter, scores, alpha=0.6, color=color, s=50, 
                  edgecolors='black', linewidth=0.5, label=strategy)
        
        # Add mean line
        mean_score = scores.mean()
        ax.axhline(mean_score, color=color, linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.set_ylabel('Retrieval Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved scatter plot to {output_path}")


def create_category_comparison_bar(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Performance by Query Category"
) -> None:
    """Create a grouped bar chart comparing strategies across query categories."""
    if 'category' not in df.columns or 'strategy' not in df.columns or 'avg_score' not in df.columns:
        print("⚠️ Required columns (category, strategy, avg_score) not found")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    category_perf = df.groupby(['strategy', 'category'])['avg_score'].mean().unstack(fill_value=0)
    
    x = np.arange(len(category_perf.index))
    width = 0.8 / len(category_perf.columns)
    
    colors = sns.color_palette("husl", len(category_perf.columns))
    
    for idx, (category, color) in enumerate(zip(category_perf.columns, colors)):
        offset = (idx - len(category_perf.columns)/2 + 0.5) * width
        bars = ax.bar(x + offset, category_perf[category], width, 
                     label=category, alpha=0.8, color=color, 
                     edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(category_perf.index, rotation=45, ha='right')
    ax.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved category comparison to {output_path}")


def create_namespace_accuracy_bar(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Namespace Detection Accuracy"
) -> None:
    """Create a bar chart showing namespace detection accuracy by strategy."""
    if 'namespace_correct' not in df.columns or 'strategy' not in df.columns:
        print("⚠️ Required columns (namespace_correct, strategy) not found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    namespace_acc = df.groupby('strategy')['namespace_correct'].mean().sort_values(ascending=False)
    
    bars = ax.bar(namespace_acc.index, namespace_acc.values,
                 color=sns.color_palette("husl", len(namespace_acc)), 
                 alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage labels
    for bar, val in zip(bars, namespace_acc.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved namespace accuracy chart to {output_path}")


def create_baseline_improvement_bar(
    improvements_df: pd.DataFrame,
    output_path: Path,
    title: str = "Improvement Over Baselines"
) -> None:
    """Create a bar chart showing improvements over baselines."""
    if improvements_df is None or len(improvements_df) == 0:
        print("⚠️ No baseline improvement data available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Group by main strategy and calculate average improvement
    main_strategies = improvements_df['main_strategy'].unique()
    improvement_means = []
    
    for strat in main_strategies:
        strat_data = improvements_df[improvements_df['main_strategy'] == strat]
        avg_improvement = strat_data['mean_improvement_pct'].mean()
        improvement_means.append({'strategy': strat, 'improvement': avg_improvement})
    
    imp_df = pd.DataFrame(improvement_means).sort_values('improvement', ascending=False)
    
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in imp_df['improvement']]
    bars = ax.barh(imp_df['strategy'], imp_df['improvement'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Average Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars, imp_df['improvement']):
        width = bar.get_width()
        ax.text(width + (2 if val >= 0 else -2), bar.get_y() + bar.get_height()/2,
               f'{val:.1f}%', ha='left' if val >= 0 else 'right', 
               va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved baseline improvement chart to {output_path}")


def create_all_basic_visualizations(
    data: Dict,
    output_dir: Path
) -> None:
    """Create all basic visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = data["results"]
    strategy_stats = data.get("strategy_stats")
    
    # Convert to DataFrame if needed
    if strategy_stats is not None:
        if isinstance(strategy_stats, list):
            strategy_stats = pd.DataFrame(strategy_stats)
        elif not isinstance(strategy_stats, pd.DataFrame):
            strategy_stats = pd.DataFrame([strategy_stats])
    
    # 1. Bar chart - Average scores
    if strategy_stats is not None and len(strategy_stats) > 0:
        create_strategy_comparison_bar_chart(
            strategy_stats,
            output_dir / '1_strategy_comparison_bar.png',
            "Strategy Comparison - Average Scores"
        )
    
    # 2. Box plot - Score distributions
    create_score_distribution_boxplot(
        df,
        output_dir / '2_score_distribution_boxplot.png',
        "Score Distribution by Strategy (Box Plot)"
    )
    
    # 3. Histogram - Score distributions
    create_score_histogram(
        df,
        output_dir / '3_score_distribution_histogram.png',
        "Score Distribution Histogram by Strategy"
    )
    
    # 4. Scatter plot - Individual scores
    create_score_scatter_plot(
        df,
        output_dir / '4_score_scatter_plot.png',
        "Individual Query Scores by Strategy"
    )
    
    # 5. Category comparison
    if 'category' in df.columns:
        create_category_comparison_bar(
            df,
            output_dir / '5_category_comparison.png',
            "Performance by Query Category"
        )
    
    # 6. Namespace accuracy
    if 'namespace_correct' in df.columns:
        create_namespace_accuracy_bar(
            df,
            output_dir / '6_namespace_accuracy.png',
            "Namespace Detection Accuracy"
        )
    
    # 7. Baseline improvements
    improvements = data.get("improvements")
    if improvements is not None:
        improvements_df = pd.DataFrame(improvements) if isinstance(improvements, list) else improvements
        if len(improvements_df) > 0:
            create_baseline_improvement_bar(
                improvements_df,
                output_dir / '7_baseline_improvements.png',
                "Improvement Over Baselines"
            )
    
    print(f"\n✅ All basic visualizations saved to {output_dir}")
