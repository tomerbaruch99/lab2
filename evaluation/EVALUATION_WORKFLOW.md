# Complete Evaluation Workflow Guide

This document provides a clear, step-by-step guide for evaluating the RAG system. The workflow is organized into two main phases:

1. **PHASE 1: Find Best Configuration** - Test strategies, chunk configs, K values, and baselines
2. **PHASE 2: Test Query Enrichment & Reranking** - Evaluate enhancements on your best configuration

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
   
   **For baseline methods** (required for Phase 1):
   ```bash
   pip install scikit-learn>=1.0.0
   ```

---

## Complete Evaluation Workflow

```
┌─────────────────────────────────────────────────────────────┐
│              COMPLETE EVALUATION WORKFLOW                   │
└─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════
PHASE 1: FIND BEST CONFIGURATION
═══════════════════════════════════════════════════════════════
Goal: Determine optimal strategy, chunk config, and K value

Step 1.1: Test Chunking Strategies
  └─> Compare baseline, sentence, adaptive strategies

Step 1.2: Test Chunk Configurations  
  └─> Compare different chunk sizes and overlaps

Step 1.3: Test K Values
  └─> Test different numbers of retrieved chunks (K=3, 5, 10, etc.)

Step 1.4: Compare Against Baselines
  └─> Validate improvements over TF-IDF, keyword matching, etc.

Step 1.5: Analyze Results & Select Best Configuration
  └─> Review all results and identify best performing combination
      (strategy + chunk_config + K value)

═══════════════════════════════════════════════════════════════
PHASE 2: TEST QUERY ENRICHMENT & RERANKING
═══════════════════════════════════════════════════════════════
Goal: Evaluate enhancements on your best configuration

Step 2.1: Run Enrichment & Reranking Evaluation
  └─> Test all 4 combinations:
      - Baseline (no enrichment, no reranking)
      - Enrichment only
      - Reranking only  
      - Both enrichment and reranking

Step 2.2: Analyze Results
  └─> Compare improvements and determine if enhancements are worth it

═══════════════════════════════════════════════════════════════
```

**IMPORTANT**: Complete Phase 1 first to identify your best configuration, then use that configuration in Phase 2.

---

## Quick Start Guide

**New to evaluation?** Follow these steps in order:

### Phase 1: Find Best Configuration (Required First)

1. **Test Strategies** (Step 1.1) - Find best chunking strategy
2. **Test Chunk Configs** (Step 1.2) - Find best chunk size/overlap (optional)
3. **Test K Values** (Step 1.3) - Find best number of chunks to retrieve
4. **Compare Baselines** (Step 1.4) - Validate improvements
5. **Analyze & Select** (Step 1.5) - Choose your best configuration

**Output**: Best strategy, chunk config, and K value

### Phase 2: Test Enhancements (After Phase 1)

1. **Run Enrichment/Reranking** (Step 2.1) - Test all 4 combinations
2. **Analyze Results** (Step 2.2) - Decide if enhancements are worth it

**Output**: Decision on whether to use enrichment, reranking, both, or neither

---

# PHASE 1: FIND BEST CONFIGURATION

This phase systematically tests different configurations to find the best performing combination of:
- **Strategy**: baseline, sentence, or adaptive chunking
- **Chunk Configuration**: chunk size and overlap
- **K Value**: number of chunks to retrieve
- **Baseline Comparison**: validate improvements over traditional methods

## Step 1.1: Test Chunking Strategies

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
    --top_k 3 \
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

## Step 1.2: Test Chunk Configurations

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
        "description": "Medium chunks (default configuration)"
    },
    {
        "name": "large_chunks",
        "chunk_chars": 2000,
        "chunk_overlap": 400,
        "description": "Large chunks for broader context"
    },
    {
        "name": "small_overlap",
        "chunk_chars": 1000,
        "chunk_overlap": 50,
        "description": "Medium chunks with minimal overlap"
    },
    {
        "name": "large_overlap",
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
    --top_k 3
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

## Step 1.3: Test K Values (Top-K)

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
for k in 3 5 10; do
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

## Step 1.4: Compare Against Baselines

**Goal**: Demonstrate system improvements over traditional retrieval methods.

### Ensure Baseline Dependencies

```bash
# Install scikit-learn for baseline methods (scipy is already in requirements.txt)
pip install scikit-learn>=1.0.0
```

### Run Evaluation with Baselines

```bash
python evaluation/generate_evaluation_results.py \
    --strategies baseline sentence adaptive \
    --top_k 3 \
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

### Review Baseline Comparisons

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

## Step 1.5: Analyze Results & Select Best Configuration

**Goal**: Review all evaluation results and identify the best performing configuration.

### Review Results from All Steps

After completing Steps 1.1-1.4, you should have results for:
- Different strategies (baseline, sentence, adaptive)
- Different chunk configurations (various sizes/overlaps)
- Different K values (3, 5, 10, etc.)
- Baseline comparisons

### Analyze Results

**Option A: Command-Line Analysis**
```bash
# Analyze results from each evaluation
python evaluation/analyze_results.py \
    --results_dir evaluation/evaluation_results
```

**Option B: Jupyter Notebook (Recommended)**
```bash
# Open Jupyter notebook for visualizations
jupyter notebook evaluation/analyze_evaluation_results.ipynb
```

### Key Files to Review

1. **`strategy_statistics.csv`** - Compare mean scores across strategies
2. **`config_comparison.csv`** - Compare chunk configurations
3. **`improvements_over_baselines.csv`** - Validate improvements
4. **`quantitative_analysis_report.md`** - Comprehensive analysis

### Select Best Configuration

Based on your analysis, identify:
- **Best Strategy**: baseline, sentence, or adaptive
- **Best Chunk Config**: chunk size and overlap (or use default if not testing configs)
- **Best K Value**: optimal number of chunks to retrieve

**Example Decision**:
```
Best Strategy: adaptive
Best Chunk Config: medium_chunks (1000 chars, 150 overlap)
Best K Value: 5
```

**Document your selection** - You'll need these values for Phase 2.

---

# PHASE 2: TEST QUERY ENRICHMENT & RERANKING

**IMPORTANT**: Only proceed to Phase 2 after completing Phase 1 and identifying your best configuration.

## Step 2.1: Run Enrichment & Reranking Evaluation

**Goal**: Compare all 4 combinations of query enrichment and reranking against your best configuration.

### Overview

The system supports two optional enhancements:
- **Query Enrichment**: Expands queries with additional keywords using LLM to improve retrieval
- **Reranking**: Uses LLM to rerank retrieved chunks by relevance before generating answers

This evaluation tests all 4 combinations:
1. **Baseline**: No enrichment, no reranking (your best config from Phase 1)
2. **Enrichment Only**: Query enrichment enabled, reranking disabled
3. **Reranking Only**: Query enrichment disabled, reranking enabled
4. **Both**: Both enrichment and reranking enabled

### Run LLM Judge Evaluation with Enrichment/Reranking

Use your best configuration from Phase 1. Replace the example values below with your actual best configuration:

```bash
# Example: Using best config from Phase 1
# Replace with YOUR best configuration values:
#   --strategies <your_best_strategy>
#   --top_k <your_best_k>
#   --index_name <your_best_chunk_config_index>  # if you tested chunk configs

python evaluation/run_llm_judge_evaluation.py \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/llm_judge_results \
    --strategies adaptive \
    --top_k 3 \
    --index_name haifa-municipality-rag-small-chunks \
    --test_enrichment_reranking \
    --baseline_strategy adaptive  # Use your best performing strategy from Phase 1
```

**What This Does**:
- Tests your best strategy from Phase 1
- Runs evaluation with all 4 combinations of enrichment/reranking:
  - Baseline (no enrichment, no reranking)
  - Enrichment only
  - Reranking only
  - Both enrichment and reranking
- Compares each combination against the baseline
- Generates comparison statistics showing improvements

**Important Notes**:
- Use your **best strategy** from Phase 1 in `--strategies` and `--baseline_strategy`
- Use your **best K value** from Phase 1 in `--top_k`
- If you tested chunk configurations, use your **best chunk config index** in `--index_name`

**Output Files**:
- `llm_judge_results.csv` - Raw results for all configurations
- `llm_judge_statistics.csv` - Aggregate statistics per configuration
- `enrichment_reranking_comparison.csv` - Detailed comparison against baseline

---

## Step 2.2: Analyze Enrichment & Reranking Results

**Goal**: Compare improvements and determine if enhancements are worth the added cost/latency.

### Review Results

The comparison CSV shows for each metric (correctness, faithfulness, completeness, conciseness, overall):
- **Mean Improvement**: Average improvement over baseline
- **Median Improvement**: Median improvement over baseline
- **Proportion Better**: Percentage of queries performing better than baseline

```bash
# View comparison results
cat evaluation/llm_judge_results/enrichment_reranking_comparison.csv

# View overall statistics
cat evaluation/llm_judge_results/llm_judge_statistics.csv
```

### Key Metrics to Look For

- **Overall Score Improvement**: Which combination performs best?
- **Consistency**: Are improvements consistent across queries? (high proportion better)
- **Trade-offs**: Consider cost and latency vs. performance gains

### Interpreting Results

- **"both" shows highest improvement** → Both features work well together
- **"enrichment_only" shows best results** → Reranking may not be necessary
- **"reranking_only" shows best results** → Query enrichment may not help
- **Baseline is best** → Enhancements may not be worth the added cost/latency

### Decision Making

After reviewing results, decide:
1. **Which combination performs best?**
2. **Is the improvement worth the added cost/latency?**
3. **Should you deploy with enrichment, reranking, both, or neither?**

---

# Additional Scenarios

## Scenario: Comprehensive Combined Evaluation

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
            --index_name "haifa-municipality-rag-${config//_/-}" \
            --output_dir "evaluation/eval_results_per_config/${config}_k${k}" \
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
    --index_name haifa-municipality-rag-medium-chunks \
    --include_baselines \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/eval_results_per_config/final
```

---

## Complete Step-by-Step Example

Here's a complete example following the two-phase workflow:

### Initial Setup (One-time)

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

### Phase 1: Find Best Configuration

```bash
# Step 1.1: Test strategies
python evaluation/generate_evaluation_results.py \
    --strategies baseline sentence adaptive \
    --top_k 5 \
    --testset_file tests/embedding_testset.json

# Step 1.2: Test chunk configurations (optional)
python evaluation/evaluate_chunk_configurations.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --configs evaluation/chunk_configs.json \
    --testset_file tests/embedding_testset.json

# Step 1.3: Test K values
for k in 3 5 10; do
    python evaluation/generate_evaluation_results.py \
        --strategies adaptive \
        --top_k $k \
        --testset_file tests/embedding_testset.json \
        --output_dir evaluation/results_k${k}
done

# Step 1.4: Compare against baselines
python evaluation/generate_evaluation_results.py \
    --strategies baseline sentence adaptive \
    --top_k 5 \
    --include_baselines \
    --testset_file tests/embedding_testset.json

# Step 1.5: Analyze and select best configuration
jupyter notebook evaluation/analyze_evaluation_results.ipynb
# Review results and document: best_strategy, best_chunk_config, best_k
```

### Phase 2: Test Query Enrichment & Reranking

```bash
# Step 2.1: Run enrichment/reranking evaluation
# (Using best config from Phase 1: adaptive strategy, K=5)
python evaluation/run_llm_judge_evaluation.py \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/llm_judge_results \
    --strategies adaptive \
    --top_k 5 \
    --test_enrichment_reranking \
    --baseline_strategy adaptive

# Step 2.2: Analyze results
cat evaluation/llm_judge_results/enrichment_reranking_comparison.csv
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

### Phase 1 Best Practices

1. **Follow the Steps in Order**: Test strategies → chunk configs → K values → baselines
2. **Use Ground Truth**: Always use `tests/embedding_testset.json` for robust metrics
3. **Include Baselines**: Essential for demonstrating system effectiveness
4. **Test Multiple K Values**: Different queries may need different numbers of chunks
5. **Compare Systematically**: Use the same queries for fair comparison
6. **Document Your Results**: Keep track of what you tested and the results

### Phase 2 Best Practices

1. **Use Your Best Config**: Only test enrichment/reranking on your best Phase 1 configuration
2. **Consider Cost vs. Benefit**: Enrichment and reranking add API calls (cost and latency)
3. **Test on Best K Value**: Reranking is more effective with more candidates (higher top_k)
4. **Make Informed Decisions**: Compare improvements against added cost/latency

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

### Phase 1: Find Best Configuration

**Test Strategies:**
```bash
python evaluation/generate_evaluation_results.py \
    --strategies baseline sentence adaptive \
    --top_k 5 \
    --testset_file tests/embedding_testset.json
```

**Test Chunk Configurations:**
```bash
python evaluation/evaluate_chunk_configurations.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --configs evaluation/chunk_configs.json \
    --testset_file tests/embedding_testset.json
```

**Test K Values:**
```bash
for k in 3 5 10; do
    python evaluation/generate_evaluation_results.py \
        --strategies adaptive \
        --top_k $k \
        --testset_file tests/embedding_testset.json \
        --output_dir "evaluation/results_k${k}"
done
```

**Compare Against Baselines:**
```bash
python evaluation/generate_evaluation_results.py \
    --strategies baseline sentence adaptive \
    --top_k 5 \
    --include_baselines \
    --testset_file tests/embedding_testset.json
```

**Analyze Results:**
```bash
jupyter notebook evaluation/analyze_evaluation_results.ipynb
```

### Phase 2: Test Query Enrichment & Reranking

**Run Enrichment/Reranking Evaluation:**
```bash
# Use your best config from Phase 1
python evaluation/run_llm_judge_evaluation.py \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/llm_judge_results \
    --strategies <your_best_strategy> \
    --top_k <your_best_k> \
    --test_enrichment_reranking \
    --baseline_strategy <your_best_strategy>
```

**Review Results:**
```bash
cat evaluation/llm_judge_results/enrichment_reranking_comparison.csv
```
