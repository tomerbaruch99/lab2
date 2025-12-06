# Evaluation Folder

This folder contains all files related to evaluating chunking strategies for the RAG system.

## Files

### Main Notebook
- **`evaluate_chunking_strategies.ipynb`** - Interactive Jupyter notebook for running evaluations with inline plots
  - Run this notebook to evaluate all chunking strategies
  - All visualizations are displayed inline for easy viewing
  - Results are saved to `results/` folder

### Scripts
- **`evaluate_chunking_strategies.py`** - Python script version (can be run from command line)
  - Use this for automated/batch evaluation
  - Same functionality as the notebook but non-interactive

### Data
- **`evaluation_queries.json`** - Set of Hebrew queries for evaluation
  - Contains 20 queries covering different namespaces and categories
  - Can be customized for your evaluation needs

### Documentation
- **`EVALUATION_README.md`** - Detailed usage guide for the evaluation system
- **`EVALUATION_SETUP_SUMMARY.md`** - Quick start summary
- **`ANALYSIS_chunking_changes.md`** - Analysis of benefits of chunking strategy changes

### Utilities
- **`create_notebook.py`** - Helper script to regenerate the notebook (if needed)

## Quick Start

### Using the Notebook (Recommended)

1. Open `evaluate_chunking_strategies.ipynb` in Jupyter
2. Run all cells to execute the evaluation
3. View inline plots and results directly in the notebook

### Using the Script

```bash
cd evaluation
python evaluate_chunking_strategies.py \
    --queries_file evaluation_queries.json \
    --output_dir ./results
```

## Output

All results are saved to the `results/` folder:
- `evaluation_results.csv` - Raw evaluation data
- `strategy_statistics.csv` - Aggregate statistics per strategy
- `namespace_statistics.csv` - Namespace detection accuracy
- `strategy_comparison.png` - Strategy comparison plots
- `namespace_accuracy_heatmap.png` - Namespace accuracy heatmap
- `category_analysis.png` - Performance by category

## Requirements

Make sure you have:
- Data indexed in Pinecone (run `indexing_2.py` first)
- `../utils/api_keys.json` with `PINECONE_API_KEY`
- Required Python packages (pandas, numpy, matplotlib, seaborn, tqdm)

## Notes

- The notebook version is recommended for interactive analysis and viewing plots
- The script version is better for automated runs or batch processing
- All plots in the notebook are displayed inline for convenience

