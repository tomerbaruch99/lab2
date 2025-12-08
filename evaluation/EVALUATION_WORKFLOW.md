# Complete Evaluation Workflow Guide

This document provides a comprehensive, step-by-step guide for evaluating the RAG system, including different chunking strategies, chunk sizes, chank overlaps, K values for retrieval, and baseline comparisons.

## Overview

The evaluation system supports multiple evaluation scenarios:

1. **Chunking Strategy Evaluation**: Compare baseline, sentence, and adaptive chunking strategies
2. **Chunk Size/Overlap Evaluation**: Test different chunk sizes and overlap configurations
3. **Top-K Value Evaluation**: Test different numbers of retrieved chunks (K)
4. **Baseline Comparison**: Compare against TF-IDF, keyword matching, and retrieval-only baselines
5. **Combined Evaluation**: Run comprehensive evaluations combining multiple factors

---

## Prerequisites

Before starting evaluation, ensure you have:

1. **Scraped Data**: `scrape_and_prepare_data/haifa_scraped.json`
   - Download from SharePoint or use the scraper notebook
   
2. **API Keys**: `utils/api_keys.json` with:
   - `PINECONE_API_KEY`
   - `GEMINI_API_KEY`

3. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   This installs all dependencies including scipy for statistical analysis.
   
   **For baseline methods** (optional, only if using `--include_baselines`):
   ```bash
   pip install scikit-learn>=1.0.0
   ```

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION WORKFLOW                       │
└─────────────────────────────────────────────────────────────┘

PHASE 1: RESULTS GENERATION (Requires API Access)
┌─────────────────────────────────────────────────────────────┐
1. PREPARE DATA (one-time or per config)
   └─> scrape_and_prepare_data/data_preparation.py

2. INDEX DATA (per configuration)
   └─> indexing.py  [Queries Pinecone API]

3. GENERATE EVALUATION RESULTS
   ├─> generate_evaluation_results.py   [Queries Pinecone/Gemini]
   │   └─> Saves CSV files locally
   └─> evaluate_chunk_configurations.py [Queries Pinecone/Gemini]
       └─> Saves CSV files locally
└─────────────────────────────────────────────────────────────┘

PHASE 2: ANALYSIS (No API Access Required)
┌─────────────────────────────────────────────────────────────┐
4. ANALYZE RESULTS (reads local CSV files only)
   ├─> analyze_results.py               [Command-line, no APIs]
   └─> analyze_evaluation_results.ipynb   [Jupyter, no APIs]
       └─> Generates visualizations and statistics
└─────────────────────────────────────────────────────────────┘
```

---

## Scenario 1: Evaluate Different Chunking Strategies

**Goal**: Compare baseline, sentence, and adaptive chunking strategies on existing indexed data.

### Step 1: Prepare Data (if not already done)

```bash
python scrape_and_prepare_data/data_preparation.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --out_dir scrape_and_prepare_data/haifa_prepared_data
```

### Step 2: Index Data (if not already done)

```bash
python indexing.py \
    --prepared_file scrape_and_prepare_data/haifa_prepared_data/haifa_rag_chunks.parquet \
    --index_name haifa-rag  # or use default
```

### Step 3: Run Evaluation

```bash
python evaluation/generate_evaluation_results.py \
    --strategies baseline sentence adaptive \
    --top_k 5 \
    --testset_file tests/embedding_testset.json  # Optional: for precision/recall metrics
```

**Output**: Results saved to `evaluation/evaluation_results/`

**Files Generated**:
- `evaluation_results.csv` - Raw results per query-strategy
- `strategy_statistics.csv` - Aggregate statistics
- `score_distribution.csv` - Query distribution across score categories
- `quantitative_analysis_report.md` - Comprehensive quantitative analysis

### Step 4: Analyze Results (No API Access Required)

After generating results, analyze them using either method:

**Option A: Command-Line Analysis**
```bash
python evaluation/analyze_results.py \
    --results_dir evaluation/evaluation_results
```

**Option B: Jupyter Notebook**
```bash
# Open Jupyter notebook
jupyter notebook evaluation/analyze_evaluation_results.ipynb
```

Both methods:
- ✅ Read only local CSV files
- ✅ Generate visualizations and statistics
- ❌ Do NOT query Pinecone or Gemini
- ❌ Do NOT require API keys
- ✅ Can run offline

---

## Scenario 2: Evaluate Different Chunk Sizes and Overlaps

**Goal**: Compare performance across different chunk size and overlap configurations.

### Step 1: Create Configuration File

Create `evaluation/chunk_configs.json`:

```json
[
    {
        "name": "small_chunks",
        "chunk_chars": 500,
        "chunk_overlap": 100,
        "description": "Small chunks for fine-grained retrieval"
    },
    {
        "name": "medium_chunks",
        "chunk_chars": 1000,
        "chunk_overlap": 150,
        "description": "Medium chunks (default)"
    },
    {
        "name": "large_chunks",
        "chunk_chars": 2000,
        "chunk_overlap": 400,
        "description": "Large chunks for broader context"
    },
    {
        "name": "high_overlap",
        "chunk_chars": 1000,
        "chunk_overlap": 300,
        "description": "Medium chunks with high overlap"
    }
]
```

### Step 2: Run Automated Evaluation

This script automatically:
1. Prepares data for each configuration
2. Indexes each configuration to separate Pinecone indexes
3. Runs evaluation on each configuration
4. Generates comparison reports

```bash
python evaluation/evaluate_chunk_configurations.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --configs evaluation/chunk_configs.json \
    --output_base_dir evaluation/chunk_config_evaluations \
    --testset_file tests/embedding_testset.json \
    --strategies baseline sentence adaptive \
    --top_k 5
```

**What Happens**:
- For each config, creates a separate Pinecone index: `haifa-municipality-rag-{config_name}`
- Prepares data with specified chunk size/overlap
- Indexes to the config-specific index
- Runs evaluation using that index
- Saves results to `evaluation/chunk_config_evaluations/{config_name}/evaluation_results/`

**Output Structure**:
```
evaluation/chunk_config_evaluations/
├── small_chunks/
│   ├── haifa_rag_chunks.parquet  (prepared data)
│   └── evaluation_results/
│       ├── evaluation_results.csv
│       ├── strategy_statistics.csv
│       └── ...
├── medium_chunks/
│   └── ...
├── large_chunks/
│   └── ...
└── config_comparison.csv  (summary comparison)
```

### Step 3: Compare Results

```bash
# Review the comparison CSV
cat evaluation/evaluation/chunk_config_evaluations/config_comparison.csv

# Or open the notebook and load multiple result directories
jupyter notebook evaluation/analyze_evaluation_results.ipynb
```

**Skip Steps** (if data/indexes already exist):
```bash
# Skip data preparation (use existing files)
python evaluation/evaluate_chunk_configurations.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --configs evaluation/chunk_configs.json \
    --skip_preparation \
    --skip_indexing  # if indexes already exist
```

---

## Scenario 3: Evaluate Different K Values (Top-K)

**Goal**: Test how many chunks to retrieve (K=3, 5, 10, etc.)

### Option A: Run Multiple Evaluations Manually

```bash
# Test K=3
python evaluation/generate_evaluation_results.py \
    --strategies adaptive \
    --top_k 3 \
    --output_dir evaluation/results_k3

# Test K=5
python evaluation/generate_evaluation_results.py \
    --strategies adaptive \
    --top_k 5 \
    --output_dir evaluation/results_k5

# Test K=10
python evaluation/generate_evaluation_results.py \
    --strategies adaptive \
    --top_k 10 \
    --output_dir evaluation/results_k10
```

### Option B: Use a Loop Script

Create `evaluation/test_top_k.sh`:

```bash
for k in 3 5 10 15; do
    echo "Testing top_k=$k"
    python evaluation/generate_evaluation_results.py \
        --strategies adaptive \
        --top_k $k \
        --output_dir "evaluation/results_k${k}" \
        --testset_file tests/embedding_testset.json
done
```

```bash
chmod +x evaluation/test_top_k.sh
./evaluation/test_top_k.sh
```

### Compare K Values

Manually compare the `strategy_statistics.csv` files from each output directory, or load them all in the notebook.

---

## Scenario 4: Compare Against Baselines

**Goal**: Demonstrate system improvements over traditional retrieval methods.

### Step 1: Ensure Baseline Dependencies

```bash
# Install scikit-learn for baseline methods (scipy is already in requirements.txt)
pip install scikit-learn>=1.0.0
```

### Step 2: Run Evaluation with Baselines

```bash
python evaluation/generate_evaluation_results.py \
    --strategies baseline sentence adaptive \
    --top_k 5 \
    --include_baselines \
    --testset_file tests/embedding_testset.json
```

**What This Adds**:
- `baseline_tfidf`: TF-IDF keyword-based retrieval
- `baseline_keyword`: Simple keyword matching
- `baseline_retrieval_only`: Semantic search without LLM generation

**Additional Output Files**:
- `improvements_over_baselines.csv` - Improvement percentages
- `statistical_significance_tests.csv` - Statistical test results (p-values)
- Enhanced `quantitative_analysis_report.md` with baseline comparisons

### Step 3: Review Baseline Comparisons

```bash
# Check improvement metrics
cat evaluation/evaluation_results/improvements_over_baselines.csv

# Check statistical significance
cat evaluation/evaluation_results/statistical_significance_tests.csv

# Read comprehensive report
cat evaluation/evaluation_results/quantitative_analysis_report.md
```

**Key Metrics to Look For**:
- **Mean Improvement**: Should be >20% over TF-IDF/keyword baselines
- **Proportion Better**: Should be >70% of queries performing better
- **Statistical Significance**: p-values < 0.05 indicate significant improvements

---

## Scenario 5: Comprehensive Combined Evaluation

**Goal**: Evaluate everything together (strategies + chunk configs + K values + baselines)

### Approach 1: Sequential Evaluation

```bash
# 1. Test different chunk configurations
python evaluation/evaluate_chunk_configurations.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --configs evaluation/chunk_configs.json \
    --testset_file tests/embedding_testset.json \
    --strategies baseline sentence adaptive

# 2. For each configuration, test different K values
for config in small_chunks medium_chunks large_chunks; do
    for k in 3 5 10; do
        python evaluation/generate_evaluation_results.py \
            --strategies adaptive \
            --top_k $k \
            --index_name "haifa-municipality-rag-${config}" \
            --output_dir "evaluation/comprehensive/${config}_k${k}" \
            --include_baselines
    done
done
```

### Approach 2: Focused Comprehensive Test

Test the best configuration with full baselines:

```bash
# Assuming medium_chunks performed best
python evaluation/generate_evaluation_results.py \
    --strategies baseline sentence adaptive \
    --top_k 5 \
    --index_name haifa-municipality-rag-medium_chunks \
    --include_baselines \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/comprehensive_final
```

---

## Complete Step-by-Step Example

Here's a complete example evaluating chunk strategies with baselines:

### 1. Initial Setup (One-time)

```bash
# Prepare data
python scrape_and_prepare_data/data_preparation.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --out_dir scrape_and_prepare_data/haifa_prepared_data

# Index data
python indexing.py \
    --prepared_file scrape_and_prepare_data/haifa_prepared_data/haifa_rag_chunks.parquet \
    --index_name haifa-rag
```

### 2. Run Evaluation

```bash
# Evaluate chunking strategies with baselines
python evaluation/generate_evaluation_results.py \
    --strategies baseline sentence adaptive \
    --top_k 5 \
    --include_baselines \
    --testset_file tests/embedding_testset.json
```

### 3. Analyze Results

```bash
# View quantitative report
cat evaluation/evaluation_results/quantitative_analysis_report.md

# Check improvements over baselines
cat evaluation/evaluation_results/improvements_over_baselines.csv

# Open notebook for visualizations
jupyter notebook evaluation/analyze_evaluation_results.ipynb
```

---

## Understanding Output Files

### Core Evaluation Files

1. **`evaluation_results.csv`**
   - One row per query-strategy combination
   - Columns: query, strategy, avg_score, max_score, namespace_correct, etc.

2. **`strategy_statistics.csv`**
   - Aggregate statistics per strategy
   - Mean, std, min, max for all metrics

3. **`score_distribution.csv`**
   - Proportion of queries in each score category
   - Excellent (≥0.8), Good (0.6-0.8), Moderate (0.4-0.6), Poor (<0.4)

### Baseline Comparison Files (if `--include_baselines` used)

4. **`improvements_over_baselines.csv`**
   - Mean/median improvement percentages
   - Number and proportion of queries performing better

5. **`statistical_significance_tests.csv`**
   - Paired t-test results
   - p-values, t-statistics, significance indicators

### Reports

6. **`quantitative_analysis_report.md`**
   - Comprehensive quantitative analysis
   - Score distributions, baseline comparisons, statistical tests
   - Interpretation and conclusions

---

## Key Evaluation Metrics

### Retrieval Quality
- **Average Score**: Mean similarity score (higher = better)
- **Score Distribution**: Proportion in each category
- **Max/Min Score**: Range of performance

### Precision/Recall (if testset used)
- **Precision**: Relevant retrieved / Total retrieved
- **Recall**: Relevant retrieved / Total relevant
- **Accuracy**: Overall correctness

### Baseline Comparisons
- **Mean Improvement**: Average % improvement over baseline
- **Proportion Better**: % of queries performing better
- **Statistical Significance**: p < 0.05 indicates significant improvement

### Namespace Accuracy
- **Detection Accuracy**: % of queries with correct namespace detection

---

## Tips for Effective Evaluation

1. **Start with Strategies**: First find the best chunking strategy
2. **Then Test Configurations**: Find optimal chunk size/overlap
3. **Refine Top-K**: Determine best number of chunks to retrieve
4. **Validate with Baselines**: Demonstrate improvements quantitatively
5. **Use Testset When Available**: Enables precision/recall metrics

### Best Practices

- **Use Ground Truth**: Always prefer `tests/embedding_testset.json` for robust metrics
- **Include Baselines**: Essential for demonstrating system effectiveness
- **Test Multiple K Values**: Different queries may need different numbers of chunks
- **Compare Systematically**: Use the same queries for fair comparison
- **Document Your Configurations**: Keep track of what you tested

---

## Troubleshooting

### Baseline Methods Not Available
```bash
# Install scikit-learn for baseline methods (scipy is already in requirements.txt)
pip install scikit-learn>=1.0.0
```

### Index Not Found
Ensure data is indexed:
```bash
python indexing.py --prepared_file <path> --index_name <name>
```

### Testset Not Found
Use default queries or create your own:
```bash
# Uses default queries if testset not found
python evaluation/generate_evaluation_results.py
```

### Out of Memory
- Test fewer configurations at once
- Reduce top_k value
- Process configurations sequentially

---

## Summary: Quick Reference

### Evaluate Strategies Only
```bash
python evaluation/generate_evaluation_results.py \
    --strategies baseline sentence adaptive
```

### Evaluate Chunk Configurations
```bash
python evaluation/evaluate_chunk_configurations.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --configs evaluation/chunk_configs.json
```

### Evaluate with Baselines
```bash
python evaluation/generate_evaluation_results.py \
    --include_baselines
```

### Evaluate Different K Values
```bash
for k in 3 5 10; do
    python evaluation/generate_evaluation_results.py \
        --top_k $k \
        --output_dir "evaluation/results_k${k}"
done
```

### Visualize Results
```bash
jupyter notebook evaluation/analyze_evaluation_results.ipynb
```

