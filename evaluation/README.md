# Evaluation System

Comprehensive evaluation framework for comparing RAG strategies, chunk configurations, and enhancement features.

## Quick Start

### Two-Phase Workflow

**Phase 1: Find Best Configuration**
```bash
# Test strategies, chunk configs, and K values
python evaluation/generate_evaluation_results.py \
    --testset_file tests/embedding_testset.json \
    --strategies baseline sentence adaptive \
    --top_k 5
```

**Phase 2: Test Enhancements** (after selecting best config)
```bash
# Test query enrichment and reranking
python evaluation/evaluate_enrichment_reranking_configs.py \
    --testset_file tests/embedding_testset.json \
    --chunk_config small_chunks --k 3 --strategy adaptive
```

**Analyze Results** (no API access needed):
```bash
jupyter notebook evaluation/analyze_evaluation_results.ipynb
```

📖 **Full workflow**: See [`EVALUATION_WORKFLOW.md`](EVALUATION_WORKFLOW.md) for complete step-by-step guide.

---

## Scripts Overview

### Results Generation

| Script | Purpose | Output |
|--------|---------|--------|
| `generate_evaluation_results.py` | Compare chunking strategies (baseline, sentence, adaptive) | CSV results with retrieval metrics |
| `evaluate_chunk_configurations.py` | Test different chunk sizes/overlaps | Comparison across chunk configs |
| `evaluate_enrichment_reranking_configs.py` | Test query enrichment & reranking combinations | LLM judge results for 4 combinations |
| `run_llm_judge_evaluation.py` | LLM-as-a-judge evaluation (answer quality) | Correctness, faithfulness, completeness scores |
| `run_comprehensive_comparison.py` | Run all combinations automatically (3 configs × 3 K × 3 strategies) | Comprehensive results in `comprehensive_comparison_results/` |

### Analysis & Visualization

| Script | Purpose | Output |
|--------|---------|--------|
| `analyze_evaluation_results.ipynb` | Interactive analysis of retrieval results | Visualizations, statistics |
| `strategy_comparison_analysis.ipynb` | Compare strategies across configs | Strategy comparison charts |
| `visualize_comprehensive_llm_judge.py` | Visualize LLM judge results | 43 PNG charts in `llm_judge_eval_results/visualizations/` |

### Utilities

- **`baseline_methods.py`** - TF-IDF, keyword matching, retrieval-only baselines
- **`llm_judge.py`** - LLM-as-a-judge for answer quality evaluation
- **`test_llm_judge.py`** - Test LLM judge functionality
- **`quantitative_report_generator.py`** - Generate analysis reports

---

## Common Use Cases

### 1. Compare Strategies

```bash
python evaluation/generate_evaluation_results.py \
    --testset_file tests/embedding_testset.json \
    --strategies baseline sentence adaptive \
    --top_k 5 \
    --include_baselines
```

**Output**: `all_strategies_comparison_eval_results/`

### 2. Test Chunk Configurations

```bash
python evaluation/evaluate_chunk_configurations.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --configs evaluation/chunk_configs.json \
    --testset_file tests/embedding_testset.json
```

**Output**: `chunks_config_comparison_eval_results/`

### 3. Test K Values

```bash
for k in 3 5 10; do
    python evaluation/generate_evaluation_results.py \
        --strategies adaptive \
        --top_k $k \
        --output_dir "evaluation/comparison_eval_results_per_k/results_k${k}"
done
```

**Output**: `comparison_eval_results_per_k/results_k{i}/`

### 4. Run Comprehensive Comparison (Optional)

**Automate all combinations** - Runs LLM judge for all chunk configs/K values/strategies at once:

```bash
python evaluation/run_comprehensive_comparison.py \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/comprehensive_comparison_results
```

**Output**: `comprehensive_comparison_results/`
- `llm_judge/` - 27 folders (one per combination)
- `baselines/` - 9 folders (baseline comparisons)
- `comparison_summary.csv` - Aggregated summary

**Use when**: You want to test all combinations automatically instead of running steps individually.

---

### 5. Evaluate Enrichment & Reranking

**Option A: Single config** (using `run_llm_judge_evaluation.py`)
```bash
python evaluation/run_llm_judge_evaluation.py \
    --testset_file tests/embedding_testset.json \
    --strategies adaptive \
    --top_k 3 \
    --test_enrichment_reranking \
    --output_dir evaluation/llm_judge_eval_results/llm_judge
```

**Output**: `llm_judge_eval_results/llm_judge/` (or custom `--output_dir`)

**Use when**: Testing a single configuration

---

**Option B: Multiple configs** (using `evaluate_enrichment_reranking_configs.py`)
```bash
# Using JSON config file
python evaluation/evaluate_enrichment_reranking_configs.py \
    --configs_file evaluation/configs_to_test.json \
    --testset_file tests/embedding_testset.json

# Using command-line arguments
python evaluation/evaluate_enrichment_reranking_configs.py \
    --chunk_config small_chunks --k 3 --strategy adaptive \
    --chunk_config small_overlap --k 10 --strategy adaptive \
    --testset_file tests/embedding_testset.json
```

**Output**: `enrichment_reranking_results/` (default, or custom `--output_dir`)

**Use when**: Testing multiple configurations at once

---

## Output Structure

### Result Folders

| Folder | Purpose | Contains |
|--------|---------|----------|
| `all_strategies_comparison_eval_results/` | Strategy comparison | CSV results, statistics, baseline comparisons |
| `chunks_config_comparison_eval_results/` | Chunk config comparison | One subfolder per config with evaluation results |
| `comparison_eval_results_per_k/` | K-value comparison | Separate folders for k=3, k=5, k=10 |
| `llm_judge_eval_results/llm_judge/` | LLM judge results | Answer quality metrics (correctness, faithfulness, etc.) |
| `comprehensive_comparison_results/` | Comprehensive comparison | All combinations (27 LLM judge + 9 baseline evaluations) |
| `enrichment_reranking_results/` | Enhancement testing | Enrichment/reranking combination results |

### Key Output Files

**From `generate_evaluation_results.py`:**
- `evaluation_results.csv` - Raw results per query-strategy
- `strategy_statistics.csv` - Aggregate statistics
- `namespace_statistics.csv` - Namespace detection accuracy
- `improvements_over_baselines.csv` - Baseline comparison (if `--include_baselines`)
- `quantitative_analysis_report.md` - Comprehensive report

**From LLM judge evaluations:**
- `llm_judge_results.csv` - Raw LLM judge scores
- `llm_judge_statistics.csv` - Aggregate statistics
- `combined_llm_judge_statistics.csv` - Combined results across configs

---

## Metrics

### Retrieval Quality
- **Average Score**: Mean similarity score (higher = better)
- **Score Distribution**: Proportion in Excellent/Good/Moderate/Poor categories
- **Precision/Recall**: With testset (ground truth labels)

### Answer Quality (LLM Judge)
- **Correctness**: Factual accuracy
- **Faithfulness**: Adherence to source material
- **Completeness**: Coverage of required information
- **Conciseness**: Brevity and clarity
- **Overall**: Combined quality score

### Baseline Comparisons
- **Mean Improvement**: Average % improvement over baselines
- **Proportion Better**: % of queries performing better
- **Statistical Significance**: p-values from paired t-tests

---

## Chunking Strategies

- **`baseline`**: Simple character-based chunking
- **`sentence`**: Sentence-aware chunking (preserves sentence boundaries)
- **`adaptive`**: Dynamic strategy selection based on document type

---

## Baseline Methods

Enable with `--include_baselines`:

- **`baseline_tfidf`**: TF-IDF keyword-based retrieval
- **`baseline_keyword`**: Simple keyword matching
- **`baseline_retrieval_only`**: Semantic search without LLM generation

**Prerequisites**: Requires `scrape_and_prepare_data/haifa_prepared_data/haifa_rag_chunks.parquet` (generate with `data_preparation.py`)

---

## Requirements

- **API Keys**: `utils/api_keys.json` with `PINECONE_API_KEY` and `GEMINI_API_KEY`
- **Dependencies**: `pip install -r requirements.txt`
- **Baseline Methods**: `pip install scikit-learn>=1.0.0`
- **Indexed Data**: Pinecone index with your data

---

## Troubleshooting

**Baseline methods return zero scores**: Generate the parquet file:
```bash
python scrape_and_prepare_data/data_preparation.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --out_dir scrape_and_prepare_data/haifa_prepared_data
```

**No results**: Check that data is indexed and index name matches

**Low scores**: Verify embedding model matches, queries are in Hebrew, data quality

**Namespace issues**: Review detection rules in `retriever.py`

---

## Documentation

- **[EVALUATION_WORKFLOW.md](EVALUATION_WORKFLOW.md)** - Complete step-by-step workflow guide
- **[LLM_JUDGE_USAGE.md](LLM_JUDGE_USAGE.md)** - LLM-as-a-judge evaluation guide
