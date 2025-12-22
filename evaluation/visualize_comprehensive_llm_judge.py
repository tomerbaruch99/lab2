"""
Comprehensive LLM Judge Results Visualization
=============================================

This script creates comprehensive visualizations for LLM judge evaluation results,
addressing all comparisons:
- Different K values (3, 5, 10)
- Different chunk sizes (small_chunks, medium_chunks, small_overlap)
- Retrieval strategies (baseline, sentence, adaptive)
- Multiple metrics (correctness, faithfulness, completeness, conciseness, overall)

Usage:
    python evaluation/visualize_comprehensive_llm_judge.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configuration
RESULTS_DIR = Path(__file__).parent / "llm_judge_eval_results" / "llm_judge"
OUTPUT_DIR = Path(__file__).parent / "llm_judge_eval_results" / "visualizations"
METRICS = ['correctness', 'faithfulness', 'completeness', 'conciseness', 'overall']
STRATEGIES = ['baseline', 'sentence', 'adaptive']
K_VALUES = [3, 5, 10]
CHUNK_CONFIGS = ['small_chunks', 'medium_chunks', 'small_overlap']


def load_comprehensive_data():
    """Load the combined LLM judge statistics."""
    stats_file = RESULTS_DIR / "combined_llm_judge_statistics.csv"
    if not stats_file.exists():
        raise FileNotFoundError(f"Combined statistics file not found: {stats_file}")
    
    df = pd.read_csv(stats_file, encoding="utf-8")
    print(f"✅ Loaded {len(df)} configurations from {stats_file}")
    return df


def create_strategy_comparison_by_k(df, metric='overall', output_path=None):
    """Compare strategies across different K values."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, k in enumerate(K_VALUES):
        ax = axes[idx]
        k_data = df[df['k'] == k]
        
        strategies_data = []
        for strategy in STRATEGIES:
            strat_data = k_data[k_data['strategy'] == strategy]
            if len(strat_data) > 0:
                mean_col = f'{metric}_mean'
                std_col = f'{metric}_std'
                strategies_data.append({
                    'strategy': strategy,
                    'mean': strat_data[mean_col].mean(),
                    'std': strat_data[mean_col].std()
                })
        
        if strategies_data:
            strat_df = pd.DataFrame(strategies_data)
            x = np.arange(len(strat_df))
            bars = ax.bar(x, strat_df['mean'], yerr=strat_df['std'], 
                         capsize=8, alpha=0.8, edgecolor='black', linewidth=1.5,
                         color=sns.color_palette("husl", len(strat_df)))
            
            ax.set_xticks(x)
            ax.set_xticklabels(strat_df['strategy'], fontsize=11, fontweight='bold')
            ax.set_ylabel(f'{metric.title()} Score', fontsize=12, fontweight='bold')
            ax.set_title(f'K={k}', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(bottom=0)
            
            # Add value labels
            for bar, (_, row) in zip(bars, strat_df.iterrows()):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + row['std'] + 0.005,
                       f'{row["mean"]:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'{metric.title()} by Strategy Across K Values', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved strategy comparison by K to {output_path}")
    plt.close()


def create_k_value_comparison(df, metric='overall', output_path=None):
    """Compare performance across different K values."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, strategy in enumerate(STRATEGIES):
        ax = axes[idx]
        strat_data = df[df['strategy'] == strategy]
        
        k_means = []
        k_stds = []
        for k in K_VALUES:
            k_data = strat_data[strat_data['k'] == k]
            if len(k_data) > 0:
                mean_col = f'{metric}_mean'
                std_col = f'{metric}_std'
                k_means.append(k_data[mean_col].mean())
                k_stds.append(k_data[mean_col].std())
            else:
                k_means.append(0)
                k_stds.append(0)
        
        bars = ax.bar(range(len(K_VALUES)), k_means, yerr=k_stds, 
                     capsize=8, alpha=0.8, edgecolor='black', linewidth=1.5,
                     color=sns.color_palette("husl", len(K_VALUES)))
        
        ax.set_xticks(range(len(K_VALUES)))
        ax.set_xticklabels([f'K={k}' for k in K_VALUES], fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{metric.title()} Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{strategy.title()} Strategy', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Add value labels
        for bar, mean, std in zip(bars, k_means, k_stds):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'{metric.title()} Across K Values by Strategy', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved K value comparison to {output_path}")
    plt.close()


def create_chunk_size_comparison(df, metric='overall', output_path=None):
    """Compare performance across different chunk configurations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    chunk_labels = {
        'small_chunks': 'Small Chunks',
        'medium_chunks': 'Medium Chunks',
        'small_overlap': 'Small Overlap'
    }
    
    for idx, strategy in enumerate(STRATEGIES):
        ax = axes[idx]
        strat_data = df[df['strategy'] == strategy]
        
        chunk_means = []
        chunk_stds = []
        chunk_names = []
        for chunk_config in CHUNK_CONFIGS:
            chunk_data = strat_data[strat_data['chunk_config'] == chunk_config]
            if len(chunk_data) > 0:
                mean_col = f'{metric}_mean'
                std_col = f'{metric}_std'
                chunk_means.append(chunk_data[mean_col].mean())
                chunk_stds.append(chunk_data[mean_col].std())
                chunk_names.append(chunk_labels.get(chunk_config, chunk_config))
        
        if chunk_means:
            x = np.arange(len(chunk_names))
            bars = ax.bar(x, chunk_means, yerr=chunk_stds, 
                         capsize=8, alpha=0.8, edgecolor='black', linewidth=1.5,
                         color=sns.color_palette("husl", len(chunk_names)))
            
            ax.set_xticks(x)
            ax.set_xticklabels(chunk_names, fontsize=10, fontweight='bold', rotation=15, ha='right')
            ax.set_ylabel(f'{metric.title()} Score', fontsize=12, fontweight='bold')
            ax.set_title(f'{strategy.title()} Strategy', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(bottom=0)
            
            # Add value labels
            for bar, mean, std in zip(bars, chunk_means, chunk_stds):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                           f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'{metric.title()} Across Chunk Configurations by Strategy', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved chunk size comparison to {output_path}")
    plt.close()


def create_comprehensive_heatmap(df, metric='overall', output_path=None):
    """Create a comprehensive heatmap showing all combinations."""
    # Create a pivot table
    pivot_data = []
    for chunk_config in CHUNK_CONFIGS:
        for k in K_VALUES:
            for strategy in STRATEGIES:
                subset = df[(df['chunk_config'] == chunk_config) & 
                           (df['k'] == k) & 
                           (df['strategy'] == strategy)]
                if len(subset) > 0:
                    mean_col = f'{metric}_mean'
                    pivot_data.append({
                        'chunk_config': chunk_config,
                        'k': k,
                        'strategy': strategy,
                        'score': subset[mean_col].iloc[0]
                    })
    
    pivot_df = pd.DataFrame(pivot_data)
    
    # Create subplots for each chunk config
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    chunk_labels = {
        'small_chunks': 'Small Chunks',
        'medium_chunks': 'Medium Chunks',
        'small_overlap': 'Small Overlap'
    }
    
    for idx, chunk_config in enumerate(CHUNK_CONFIGS):
        ax = axes[idx]
        chunk_data = pivot_df[pivot_df['chunk_config'] == chunk_config]
        
        # Create pivot for heatmap
        heatmap_data = chunk_data.pivot(index='strategy', columns='k', values='score')
        # Reindex to ensure all strategies and K values are present (fill missing with NaN)
        heatmap_data = heatmap_data.reindex(STRATEGIES)
        heatmap_data = heatmap_data.reindex(columns=K_VALUES)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'shrink': 0.8, 'label': f'{metric.title()} Score'}, 
                   ax=ax, vmin=0, vmax=1, linewidths=1, linecolor='black')
        
        ax.set_title(chunk_labels.get(chunk_config, chunk_config), 
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('K Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Strategy', fontsize=11, fontweight='bold')
    
    plt.suptitle(f'{metric.title()} Heatmap: Strategy × K Value × Chunk Config', 
                fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comprehensive heatmap to {output_path}")
    plt.close()


def create_all_metrics_comparison(df, output_path=None):
    """Compare all metrics side by side."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        
        # Aggregate by strategy
        strategy_means = []
        for strategy in STRATEGIES:
            strat_data = df[df['strategy'] == strategy]
            if len(strat_data) > 0:
                mean_col = f'{metric}_mean'
                strategy_means.append(strat_data[mean_col].mean())
            else:
                strategy_means.append(0)
        
        bars = ax.bar(STRATEGIES, strategy_means, alpha=0.8, 
                     edgecolor='black', linewidth=1.5,
                     color=sns.color_palette("husl", len(STRATEGIES)))
        
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.title()}', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0, top=1)
        
        # Add value labels
        for bar, mean in zip(bars, strategy_means):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Hide the last subplot
    axes[-1].axis('off')
    
    plt.suptitle('All Metrics Comparison by Strategy', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved all metrics comparison to {output_path}")
    plt.close()


def create_line_plot_k_vs_metric(df, metric='overall', output_path=None):
    """Create line plots showing how metrics change with K values."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    chunk_labels = {
        'small_chunks': 'Small Chunks',
        'medium_chunks': 'Medium Chunks',
        'small_overlap': 'Small Overlap'
    }
    
    colors = sns.color_palette("husl", len(STRATEGIES) * len(CHUNK_CONFIGS))
    color_idx = 0
    
    for strategy in STRATEGIES:
        for chunk_config in CHUNK_CONFIGS:
            chunk_strat_data = df[(df['strategy'] == strategy) & 
                                 (df['chunk_config'] == chunk_config)]
            
            k_values_sorted = []
            means = []
            
            for k in K_VALUES:
                k_data = chunk_strat_data[chunk_strat_data['k'] == k]
                if len(k_data) > 0:
                    mean_col = f'{metric}_mean'
                    k_values_sorted.append(k)
                    means.append(k_data[mean_col].iloc[0])
            
            if len(k_values_sorted) > 0:
                label = f'{strategy.title()} - {chunk_labels.get(chunk_config, chunk_config)}'
                ax.plot(k_values_sorted, means, marker='o', linewidth=2.5, 
                       markersize=10, label=label, color=colors[color_idx],
                       alpha=0.8, markeredgecolor='black', markeredgewidth=1.5)
                color_idx += 1
    
    ax.set_xlabel('K Value', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.title()} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric.title()} vs K Value: All Configurations', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved line plot to {output_path}")
    plt.close()


def create_best_configuration_analysis(df, output_path=None):
    """Analyze and visualize the best configurations."""
    results = []
    
    for metric in METRICS:
        mean_col = f'{metric}_mean'
        best_row = df.loc[df[mean_col].idxmax()]
        results.append({
            'metric': metric,
            'best_score': best_row[mean_col],
            'chunk_config': best_row['chunk_config'],
            'k': best_row['k'],
            'strategy': best_row['strategy']
        })
    
    results_df = pd.DataFrame(results)
    
    # Create bar chart showing best scores for each metric
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(METRICS))
    bars = ax.bar(x, results_df['best_score'], alpha=0.8, 
                 edgecolor='black', linewidth=1.5,
                 color=sns.color_palette("husl", len(METRICS)))
    
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Score', fontsize=12, fontweight='bold')
    ax.set_title('Best Score Achieved for Each Metric', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in METRICS], fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    # Add value labels with configuration info
    for i, (bar, (_, row)) in enumerate(zip(bars, results_df.iterrows())):
        height = bar.get_height()
        config_label = f"{row['strategy']}, K={row['k']}\n{row['chunk_config'].replace('_', ' ')}"
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}\n({config_label})', ha='center', va='bottom', 
               fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved best configuration analysis to {output_path}")
    plt.close()
    
    # Also create a summary table visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            row['metric'].title(),
            f"{row['best_score']:.4f}",
            row['chunk_config'].replace('_', ' ').title(),
            f"K={row['k']}",
            row['strategy'].title()
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Metric', 'Best Score', 'Chunk Config', 'K', 'Strategy'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2, 0.2, 0.25, 0.15, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Best Configuration Summary', fontsize=14, fontweight='bold', pad=20)
    
    if output_path:
        summary_path = output_path.parent / f"{output_path.stem}_summary.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved best configuration summary to {summary_path}")
    plt.close()


def create_strategy_vs_chunk_heatmap(df, metric='overall', k_value=None, output_path=None):
    """Create heatmap comparing strategies vs chunk configs for a specific K."""
    if k_value is None:
        k_value = 5  # Default to K=5
    
    k_data = df[df['k'] == k_value]
    
    # Create pivot table
    pivot_data = []
    for chunk_config in CHUNK_CONFIGS:
        for strategy in STRATEGIES:
            subset = k_data[(k_data['chunk_config'] == chunk_config) & 
                           (k_data['strategy'] == strategy)]
            if len(subset) > 0:
                mean_col = f'{metric}_mean'
                pivot_data.append({
                    'chunk_config': chunk_config.replace('_', ' ').title(),
                    'strategy': strategy.title(),
                    'score': subset[mean_col].iloc[0]
                })
    
    pivot_df = pd.DataFrame(pivot_data)
    heatmap_data = pivot_df.pivot(index='chunk_config', columns='strategy', values='score')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
               cbar_kws={'label': f'{metric.title()} Score'}, 
               ax=ax, vmin=0, vmax=1, linewidths=1, linecolor='black')
    
    ax.set_title(f'{metric.title()} Score: Strategy × Chunk Config (K={k_value})', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Chunk Configuration', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved strategy vs chunk heatmap to {output_path}")
    plt.close()


def main():
    """Main function to create all visualizations."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE LLM JUDGE RESULTS VISUALIZATION")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📊 Saving visualizations to: {OUTPUT_DIR}\n")
    
    # Load data
    try:
        df = load_comprehensive_data()
        print(f"✅ Loaded data with columns: {', '.join(df.columns)}\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Create visualizations for each metric
    for metric in METRICS:
        print(f"\n[STEP] Creating visualizations for {metric}...")
        
        # 1. Strategy comparison by K
        create_strategy_comparison_by_k(
            df, metric=metric,
            output_path=OUTPUT_DIR / f'1_{metric}_strategy_comparison_by_k.png'
        )
        
        # 2. K value comparison
        create_k_value_comparison(
            df, metric=metric,
            output_path=OUTPUT_DIR / f'2_{metric}_k_value_comparison.png'
        )
        
        # 3. Chunk size comparison
        create_chunk_size_comparison(
            df, metric=metric,
            output_path=OUTPUT_DIR / f'3_{metric}_chunk_size_comparison.png'
        )
        
        # 4. Comprehensive heatmap
        create_comprehensive_heatmap(
            df, metric=metric,
            output_path=OUTPUT_DIR / f'4_{metric}_comprehensive_heatmap.png'
        )
        
        # 5. Line plot K vs metric
        create_line_plot_k_vs_metric(
            df, metric=metric,
            output_path=OUTPUT_DIR / f'5_{metric}_line_plot_k_vs_metric.png'
        )
        
        # 6. Strategy vs chunk heatmap for different K values
        for k in K_VALUES:
            create_strategy_vs_chunk_heatmap(
                df, metric=metric, k_value=k,
                output_path=OUTPUT_DIR / f'6_{metric}_strategy_vs_chunk_k{k}.png'
            )
    
    # Create overall analysis visualizations
    print(f"\n[STEP] Creating overall analysis visualizations...")
    
    # All metrics comparison
    create_all_metrics_comparison(
        df,
        output_path=OUTPUT_DIR / '7_all_metrics_comparison.png'
    )
    
    # Best configuration analysis
    create_best_configuration_analysis(
        df,
        output_path=OUTPUT_DIR / '8_best_configuration_analysis.png'
    )
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\n✅ All visualizations saved to: {OUTPUT_DIR}")
    print(f"✅ Created visualizations for {len(METRICS)} metrics")
    print(f"✅ Total files created: {len(list(OUTPUT_DIR.glob('*.png')))}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

