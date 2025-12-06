# Evaluation Folder

This folder contains all files related to evaluating chunking strategies for the RAG system.

## Files

### Main Notebook
- **`evaluate_chunking_strategies.ipynb`** - Interactive Jupyter notebook for visualization and analysis
  - Loads CSV results from the Python script
  - Generates all visualizations inline for easy viewing
  - Provides detailed analysis and comparisons

### Scripts
- **`evaluate_chunking_strategies.py`** - Python script for running evaluation
  - Runs evaluation across all strategies
  - Saves results to CSV files (no visualizations)
  - Use for automated/batch evaluation

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

### Step 1: Run Evaluation (Python Script)

First, run the Python script to generate CSV results:

```bash
cd evaluation
python evaluate_chunking_strategies.py \
    --queries_file evaluation_queries.json \
    --output_dir ./evaluation_results
```

### Step 2: Visualize Results (Jupyter Notebook)

Then, open the notebook to visualize and analyze:

1. Open `evaluate_chunking_strategies.ipynb` in Jupyter
2. Run all cells to load CSV results and generate visualizations
3. View inline plots and analysis directly in the notebook

## Output

### From Python Script

CSV files are saved to the output directory (default: `evaluation_results/`):
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

