# Evaluation

Evaluation system for comparing chunking strategies and analyzing RAG performance.

## Quick Start

### Two-Phase Workflow

1. **Generate Results** (requires API access):
   ```bash
   python evaluation/generate_evaluation_results.py \
       --testset_file tests/embedding_testset.json \
       --output_dir evaluation/evaluation_results
   ```

2. **Analyze Results** (no API access needed):
   ```bash
   python evaluation/analyze_results.py \
       --results_dir evaluation/evaluation_results
   ```
   Or use the Jupyter notebook: `evaluation/analyze_evaluation_results.ipynb`

## Files

### Results Generation
- **`generate_evaluation_results.py`** - Generates CSV results (queries APIs)
- **`evaluate_chunk_configurations.py`** - Tests multiple chunk size/overlap configs

### Analysis
- **`analyze_results.py`** - Command-line analysis (reads CSV only)
- **`analyze_evaluation_results.ipynb`** - Interactive notebook analysis
- **`visualize_llm_judge_results.py`** - Visualize LLM judge and enrichment/reranking results
- **`visualization_utils.py`** - Comprehensive visualization functions

### Data
- **`evaluation_queries.json`** - Hebrew evaluation queries
- **`tests/embedding_testset.json`** - Testset with ground truth labels (recommended)

### Utilities
- **`baseline_methods.py`** - Baseline methods (TF-IDF, keyword matching)
- **`llm_judge.py`** - LLM-as-a-judge for answer quality evaluation

## Usage

### Basic Evaluation

```bash
python evaluation/generate_evaluation_results.py \
    --testset_file tests/embedding_testset.json \
    --strategies baseline sentence adaptive \
    --top_k 5 \
    --output_dir evaluation/evaluation_results
```

### With Baseline Comparisons

```bash
python evaluation/generate_evaluation_results.py \
    --testset_file tests/embedding_testset.json \
    --include_baselines
```

### Test Different Chunk Configurations

**Manual method**:
1. Prepare data with specific config
2. Index to separate Pinecone index
3. Run evaluation with that index name

**Automated method**:
```bash
python evaluation/evaluate_chunk_configurations.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --configs evaluation/chunk_configs_example.json \
    --testset_file tests/embedding_testset.json
```

### Evaluate Query Enrichment and Reranking

**Important**: First determine your best configuration from standard evaluations.

```bash
# Run evaluation
python evaluation/run_llm_judge_evaluation.py \
    --testset_file tests/embedding_testset.json \
    --strategies adaptive \
    --top_k 5 \
    --test_enrichment_reranking \
    --baseline_strategy adaptive

# Visualize results
python evaluation/visualize_llm_judge_results.py \
    --results_dir evaluation/llm_judge_results
```

Tests all 4 combinations: baseline, enrichment only, reranking only, both.

## Output Files

**From generation**:
- `evaluation_results.csv` - Raw results
- `strategy_statistics.csv` - Aggregate statistics
- `namespace_statistics.csv` - Namespace detection accuracy

**From analysis**:
- `1_strategy_comparison_bar.png` - Bar chart comparing average scores
- `2_score_distribution_boxplot.png` - Box plot showing score distributions
- `3_score_distribution_histogram.png` - Histogram of score distributions by strategy
- `4_score_scatter_plot.png` - Scatter plot of individual query scores
- `5_category_comparison.png` - Grouped bar chart by query category
- `6_namespace_accuracy.png` - Bar chart of namespace detection accuracy
- `7_baseline_improvements.png` - Bar chart showing improvements over baselines (if available)
- `8_namespace_accuracy_heatmap.png` - Heatmap of namespace accuracy by namespace and strategy

## Metrics

- **Retrieval Quality**: Average similarity scores, score distribution, unique documents
- **Ground Truth** (with testset): Precision, recall, accuracy
- **Namespace Accuracy**: Detection accuracy percentage

## Chunking Strategies

- **baseline**: Simple character-based chunking
- **sentence**: Sentence-aware chunking
- **adaptive**: Dynamic strategy selection based on document type

## Baseline Methods

Include with `--include_baselines`:
- `baseline_tfidf`: TF-IDF keyword-based retrieval
- `baseline_keyword`: Simple keyword matching
- `baseline_retrieval_only`: Semantic search without LLM generation

## Requirements

- Data indexed in Pinecone
- `utils/api_keys.json` with `PINECONE_API_KEY` (and `GEMINI_API_KEY` for answer generation)
- `pip install -r requirements.txt`
- Optional: `pip install scikit-learn>=1.0.0` for baseline methods

## Troubleshooting

- **No results**: Check data is indexed, index name matches, namespace exists
- **Low scores**: Check embedding model matches, queries in Hebrew, data quality
- **Namespace issues**: Review detection rules in `retriever.py`
