# Complete Evaluation Workflow

Step-by-step guide for systematically evaluating the RAG system to find optimal configurations and test enhancements.

---

## Overview

The evaluation workflow has two phases:

1. **PHASE 1: Find Optimal Configuration** - Test chunk configs, strategies, K values, and get answer quality metrics
2. **PHASE 2: Test Query Enrichment & Reranking** - Test enhancements on your selected configuration

**Important**: Complete Phase 1 first to identify your selected configuration, then use that configuration in Phase 2.

---

## Prerequisites

Before starting evaluation:

1. **Scraped Data**: `scrape_and_prepare_data/haifa_scraped.json`
2. **API Keys**: `utils/api_keys.json` with `PINECONE_API_KEY` and `GEMINI_API_KEY`
3. **Dependencies**: 
   ```bash
   pip install -r requirements.txt
   pip install scikit-learn>=1.0.0  # For baseline methods
   ```

---

## Evaluation Folder Structure

Understanding output folders helps navigate results:

| Folder | Purpose | Use Case |
|--------|---------|----------|
| `all_strategies_comparison_eval_results/` | Compare strategies (baseline, sentence, adaptive) + baselines | Strategy comparison |
| `comparison_eval_results_per_k/results_k{i}/` | Test K values (k=3, 5, 10) | Finding optimal K |
| `chunks_config_comparison_eval_results/` | Compare chunk configs (size/overlap) | Finding optimal chunk size |
| `llm_judge_eval_results/llm_judge/` | LLM as a judge results (answer quality) | Answer quality evaluation |
| `enrichment_reranking_results/` | Test enrichment & reranking | Enhancement evaluation |

---

# PHASE 1: FIND OPTIMAL CONFIGURATION

Goal: Determine optimal strategy, chunk configuration, and K value.

## Step 1.1: Test Chunk Configurations (Optional)

**Goal**: Compare different chunk sizes and overlaps.

**Output**: `chunks_config_comparison_eval_results/`

### Create Configuration File

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

### Run Automated Evaluation

```bash
python evaluation/evaluate_chunk_configurations.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --configs evaluation/chunk_configs.json \
    --output_base_dir evaluation/chunks_config_comparison_eval_results \
    --testset_file tests/embedding_testset.json \
    --strategies baseline sentence adaptive \
    --top_k 5
```

**What Happens**:
- Creates separate Pinecone index for each config: `haifa-municipality-rag-{config_name}`
- Prepares data with specified chunk size/overlap
- Indexes to config-specific index
- Runs evaluation using that index

**Output**: `chunks_config_comparison_eval_results/{config_name}/evaluation_results/`

**Skip Steps** (if data/indexes already exist):
```bash
python evaluation/evaluate_chunk_configurations.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --configs evaluation/chunk_configs.json \
    --skip_preparation \
    --skip_indexing
```

---

## Step 1.2: Test Strategies

**Goal**: Compare baseline, sentence, and adaptive chunking strategies.

**Output**: `all_strategies_comparison_eval_results/`

**Important**: If you tested chunk configs in Step 1.1, use your selected chunk config's index. Otherwise, use your default index.

### Run Evaluation

**If you tested chunk configs** (use selected chunk config's index):
```bash
python evaluation/generate_evaluation_results.py \
    --testset_file tests/embedding_testset.json \
    --strategies baseline sentence adaptive \
    --top_k 5 \
    --index_name haifa-municipality-rag-small-chunks \
    --include_baselines \
    --output_dir evaluation/all_strategies_comparison_eval_results
```

**If you skipped chunk config testing** (use default index):
```bash
python evaluation/generate_evaluation_results.py \
    --testset_file tests/embedding_testset.json \
    --strategies baseline sentence adaptive \
    --top_k 5 \
    --include_baselines \
    --output_dir evaluation/all_strategies_comparison_eval_results
```

**Output**: `all_strategies_comparison_eval_results/`

**Files Generated**:
- `evaluation_results.csv` - Raw results per query-strategy
- `strategy_statistics.csv` - Aggregate statistics
- `improvements_over_baselines.csv` - Baseline comparisons
- `quantitative_analysis_report.md` - Comprehensive report

**Key Metrics**: Compare mean scores across strategies to identify top performer.

---

## Step 1.3: Test K Values

**Goal**: Test how many chunks to retrieve (K=3, 5, 10, etc.).

**Output**: `comparison_eval_results_per_k/results_k{i}/`

**Important**: Use your selected chunk config index (if tested) and selected strategy from Step 1.2.

### Run Multiple Evaluations

**Replace `adaptive` with your selected strategy from Step 1.2, and `haifa-municipality-rag-small-chunks` with your selected chunk config index:**

```bash
for k in 3 5 10; do
    python evaluation/generate_evaluation_results.py \
        --strategies adaptive \
        --top_k $k \
        --index_name haifa-municipality-rag-small-chunks \
        --testset_file tests/embedding_testset.json \
        --output_dir "evaluation/comparison_eval_results_per_k/results_k${k}"
done
```

**Output**: `comparison_eval_results_per_k/results_k{i}/`

**Compare Results**: Review `strategy_statistics.csv` from each folder to find optimal K.

---

## Step 1.4: Evaluate Answer Quality (LLM as a judge)

**Goal**: Get answer quality metrics (correctness, faithfulness, completeness) for your configurations.

**Output**: `llm_judge_eval_results/llm_judge/`

**Important**: This step evaluates actual answer quality, not just retrieval. Use your selected config from previous steps.

### Run LLM as a judge Evaluation

**Test multiple configs** (recommended for comprehensive comparison):
```bash
python evaluation/run_llm_judge_evaluation.py \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/llm_judge_eval_results/llm_judge \
    --strategies baseline sentence adaptive \
    --top_k 3 \
    --index_name haifa-municipality-rag-small-chunks
```

**Or test specific K values** (if you want to test K values with LLM as a judge):
```bash
for k in 3 5 10; do
    python evaluation/run_llm_judge_evaluation.py \
        --testset_file tests/embedding_testset.json \
        --output_dir evaluation/llm_judge_eval_results/llm_judge \
        --strategies adaptive \
        --top_k $k \
        --index_name haifa-municipality-rag-small-chunks
done
```

**Output**: `llm_judge_eval_results/llm_judge/{config_folder}/`

**Files Generated**:
- `llm_judge_results.csv` - Raw LLM as a judge scores per query
- `llm_judge_statistics.csv` - Aggregate statistics (correctness, faithfulness, completeness, conciseness, overall)

**Visualize Results**:
```bash
python evaluation/visualize_comprehensive_llm_judge.py
# Generates 43 PNG charts in llm_judge_eval_results/visualizations/
```

**Visualize Results**:
```bash
python evaluation/visualize_comprehensive_llm_judge.py
# Generates charts in llm_judge_eval_results/visualizations/
```

**Key Metrics**: Compare overall scores across strategies/configs/K values to identify optimal configuration.

---

## Alternative: Run Comprehensive Comparison (Optional)

**Goal**: Automate all combinations at once instead of running steps individually.

If you want to test all combinations automatically, you can use:

```bash
python evaluation/run_comprehensive_comparison.py \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/llm_judge_eval_results
```

**What This Does**:
- Runs LLM judge for all combinations
- Generates summary report: `comparison_summary.csv`

**Output**: `llm_judge_eval_results/`
- `llm_judge/{chunk_config}_k{k}_{strategy}/` - One folder per combination
- `comparison_summary.csv` - Aggregated summary (in base directory)

**Use when**: You want comprehensive results for all combinations without running steps manually.

---

## Step 1.5: Analyze Results & Select Configuration

**Goal**: Review all evaluation results and identify your selected configuration.

After completing Steps 1.1-1.4, analyze results:

```bash
# Analyze retrieval metrics
jupyter notebook evaluation/analyze_evaluation_results.ipynb

# Analyze answer quality metrics
python evaluation/visualize_comprehensive_llm_judge.py
```

### Select Configuration

Based on your analysis, identify:
- **Selected Strategy**: baseline, sentence, or adaptive (from Step 1.2)
- **Selected Chunk Config**: chunk size and overlap (from Step 1.1, or use default if skipped)
- **Selected K Value**: optimal number of chunks to retrieve (from Step 1.3)

**Consider both retrieval metrics AND answer quality metrics** when making your decision.

**Example Decision**:
```
Selected Strategy: adaptive
Selected Chunk Config: small_chunks (500 chars, 100 overlap)
Selected K Value: 3
Selected Index: haifa-municipality-rag-small-chunks
```

**Document your selection** - You'll need these values for Phase 2.

---

# PHASE 2: TEST QUERY ENRICHMENT & RERANKING

**Important**: Only proceed after completing Phase 1 and identifying your selected configuration.

## Step 2.1: Test Enrichment & Reranking

**Goal**: Compare all 4 combinations of query enrichment and reranking against your selected configuration.

**Output**: `enrichment_reranking_results/` or `llm_judge_eval_results/llm_judge/`

### Overview

The system supports two optional enhancements:
- **Query Enrichment**: Expands queries with additional keywords using LLM
- **Reranking**: Uses LLM to rerank retrieved chunks by relevance

This evaluation tests all 4 combinations:
1. **Baseline**: No enrichment, no reranking
2. **Enrichment Only**: Query enrichment enabled, reranking disabled
3. **Reranking Only**: Query enrichment disabled, reranking enabled
4. **Both**: Both enrichment and reranking enabled

### Option A: Single Configuration

Use `run_llm_judge_evaluation.py` for **one configuration**:

```bash
python evaluation/run_llm_judge_evaluation.py \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/llm_judge_eval_results/llm_judge \
    --strategies adaptive \
    --top_k 3 \
    --index_name haifa-municipality-rag-small-chunks \
    --test_enrichment_reranking \
    --baseline_strategy adaptive
```

**Replace with YOUR selected configuration values**:
- `--strategies <your_selected_strategy>`
- `--top_k <your_selected_k>`
- `--index_name <your_selected_chunk_config_index>` (if you tested chunk configs)

**Output**: `llm_judge_eval_results/llm_judge/` (or custom `--output_dir`)

**Use when**: Testing a single configuration

---

### Option B: Multiple Configurations

Use `evaluate_enrichment_reranking_configs.py` for **multiple configurations**:

**Using JSON config file**:
```bash
# Create evaluation/configs_to_test.json:
[
    {"chunk_config": "small_chunks", "k": 3, "strategy": "adaptive"},
    {"chunk_config": "small_overlap", "k": 10, "strategy": "adaptive"}
]

python evaluation/evaluate_enrichment_reranking_configs.py \
    --configs_file evaluation/configs_to_test.json \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/enrichment_reranking_results
```

**Using command-line arguments**:
```bash
python evaluation/evaluate_enrichment_reranking_configs.py \
    --chunk_config small_chunks --k 3 --strategy adaptive \
    --chunk_config small_overlap --k 10 --strategy adaptive \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/enrichment_reranking_results
```

**Output**: `enrichment_reranking_results/` (default, or custom `--output_dir`)

**Use when**: Testing multiple configurations at once (more flexible, supports JSON config files)

**Include baseline in evaluation** (don't skip it):
```bash
python evaluation/evaluate_enrichment_reranking_configs.py \
    --configs_file evaluation/configs_to_test.json \
    --include_baseline \
    --testset_file tests/embedding_testset.json
```

---

## Step 2.2: Analyze Results

**Goal**: Compare improvements and determine if enhancements are worth the added cost/latency.

### Review Results

```bash
# Review comparison results
cat evaluation/enrichment_reranking_results/aggregated_enrichment_reranking_statistics.csv

# Or for single config
cat evaluation/llm_judge_eval_results/llm_judge/enrichment_reranking_comparison.csv

# Visualize results
python evaluation/visualize_comprehensive_llm_judge.py
```

### Visualize Results

```bash
python evaluation/visualize_comprehensive_llm_judge.py
# Generates 43 PNG charts in llm_judge_eval_results/visualizations/
```

### Key Metrics

- **Overall Score Improvement**: Which combination performs highest?
- **Consistency**: Are improvements consistent across queries? (high proportion better)
- **Trade-offs**: Consider cost and latency vs. performance gains

### Interpreting Results

- **"both" shows highest improvement** → Both features work well together
- **"enrichment_only" shows highest results** → Reranking may not be necessary
- **"reranking_only" shows highest results** → Query enrichment may not help
- **Baseline performs highest** → Enhancements may not be worth the added cost/latency

### Decision Making

After reviewing results, decide:
1. **Which combination performs highest?**
2. **Is the improvement worth the added cost/latency?**
3. **Should you deploy with enrichment, reranking, both, or neither?**

---

## Complete Example Workflow

### Phase 1: Find Optimal Configuration

```bash
# Step 1.1: Test chunk configurations (optional but recommended)
# Uses: evaluate_chunk_configurations.py
python evaluation/evaluate_chunk_configurations.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --configs evaluation/chunk_configs.json \
    --testset_file tests/embedding_testset.json \
    --strategies baseline sentence adaptive \
    --top_k 5

# Step 1.2: Test strategies (use selected chunk config index from Step 1.1)
# Uses: generate_evaluation_results.py
python evaluation/generate_evaluation_results.py \
    --strategies baseline sentence adaptive \
    --top_k 5 \
    --index_name haifa-municipality-rag-small-chunks \
    --include_baselines \
    --testset_file tests/embedding_testset.json

# Step 1.3: Test K values (use selected strategy and chunk config)
# Uses: generate_evaluation_results.py
for k in 3 5 10; do
    python evaluation/generate_evaluation_results.py \
        --strategies adaptive \
        --top_k $k \
        --index_name haifa-municipality-rag-small-chunks \
        --testset_file tests/embedding_testset.json \
        --output_dir "evaluation/comparison_eval_results_per_k/results_k${k}"
done

# Step 1.4: Evaluate answer quality with LLM as a judge
# Uses: run_llm_judge_evaluation.py
python evaluation/run_llm_judge_evaluation.py \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/llm_judge_eval_results/llm_judge \
    --strategies baseline sentence adaptive \
    --top_k 3 \
    --index_name haifa-municipality-rag-small-chunks

# Step 1.5: Analyze and select configuration
jupyter notebook evaluation/analyze_evaluation_results.ipynb
python evaluation/visualize_comprehensive_llm_judge.py
# Document: selected_strategy=adaptive, selected_chunk_config=small_chunks, selected_k=3
```

### Phase 2: Test Enhancements

```bash
# Step 2.1: Run enrichment/reranking evaluation
# Option A: Single config (uses run_llm_judge_evaluation.py)
python evaluation/run_llm_judge_evaluation.py \
    --testset_file tests/embedding_testset.json \
    --strategies adaptive --top_k 3 \
    --index_name haifa-municipality-rag-small-chunks \
    --test_enrichment_reranking

# Option B: Multiple configs (uses evaluate_enrichment_reranking_configs.py)
python evaluation/evaluate_enrichment_reranking_configs.py \
    --chunk_config small_chunks --k 3 --strategy adaptive \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/enrichment_reranking_results

# Step 2.2: Analyze results
python evaluation/visualize_comprehensive_llm_judge.py
cat evaluation/enrichment_reranking_results/aggregated_enrichment_reranking_statistics.csv
```

---

## Understanding Output Files

### Core Evaluation Files

| File | Description |
|------|-------------|
| `evaluation_results.csv` | One row per query-strategy combination with retrieval scores |
| `strategy_statistics.csv` | Aggregate statistics per strategy (mean, std, min, max) |
| `score_distribution.csv` | Proportion of queries in each score category |
| `namespace_statistics.csv` | Namespace detection accuracy |

### Baseline Comparison Files

| File | Description |
|------|-------------|
| `improvements_over_baselines.csv` | Mean/median improvement percentages |
| `statistical_significance_tests.csv` | Paired t-test results (p-values, t-statistics) |

### LLM as a judge Files

| File | Description |
|------|-------------|
| `llm_judge_results.csv` | Raw LLM as a judge scores per query |
| `llm_judge_statistics.csv` | Aggregate statistics (correctness, faithfulness, etc.) |
| `combined_llm_judge_statistics.csv` | Combined results across all configs |

### Reports

| File | Description |
|------|-------------|
| `quantitative_analysis_report.md` | Comprehensive quantitative analysis with interpretations |

---

## Key Evaluation Metrics

### Retrieval Quality
- **Average Score**: Mean similarity score (higher = better)
- **Score Distribution**: Proportion in Excellent (≥0.8), Good (0.6-0.8), Moderate (0.4-0.6), Poor (<0.4)
- **Precision/Recall**: With testset (ground truth labels)

### Answer Quality (LLM as a judge)
- **Correctness**: Factual accuracy (0-1)
- **Faithfulness**: Adherence to source material (0-1)
- **Completeness**: Coverage of required information (0-1)
- **Conciseness**: Brevity and clarity (0-1)
- **Overall**: Combined quality score (0-1)

### Baseline Comparisons
- **Mean Improvement**: Average % improvement over baseline
- **Proportion Better**: % of queries performing better
- **Statistical Significance**: p < 0.05 indicates significant improvement

---

## Recommended Practices

### Phase 1
1. **Follow Steps in Order**: Test chunk configs → strategies → K values → LLM as a judge
2. **Use Ground Truth**: Always use `tests/embedding_testset.json` for robust metrics
3. **Include Baselines**: Essential for demonstrating system effectiveness
4. **Test Multiple K Values**: Different queries may need different numbers of chunks
5. **Document Results**: Keep track of what you tested and the results

### Phase 2
1. **Use Your Selected Config**: Only test enrichment/reranking on your selected Phase 1 configuration
2. **Consider Cost vs. Benefit**: Enrichment and reranking add API calls (cost and latency)
3. **Test on Selected K Value**: Reranking is more effective with more candidates (higher top_k)
4. **Make Informed Decisions**: Compare improvements against added cost/latency

---

## Troubleshooting

**Baseline methods return zero scores**: Generate the parquet file:
```bash
python scrape_and_prepare_data/data_preparation.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --out_dir scrape_and_prepare_data/haifa_prepared_data
```

**Index not found**: Ensure data is indexed:
```bash
python indexing.py --prepared_file <path> --index_name <name>
```

**Testset not found**: Scripts use default queries if testset not found

**Out of memory**: Test fewer configurations at once, reduce top_k value, process sequentially

---

## Quick Reference

### Phase 1 Commands

**Test Chunk Configurations** (do this first):
```bash
python evaluation/evaluate_chunk_configurations.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --configs evaluation/chunk_configs.json \
    --testset_file tests/embedding_testset.json \
    --strategies baseline sentence adaptive --top_k 5
```

**Test Strategies** (use selected chunk config index):
```bash
python evaluation/generate_evaluation_results.py \
    --strategies baseline sentence adaptive --top_k 5 \
    --index_name haifa-municipality-rag-small-chunks \
    --include_baselines --testset_file tests/embedding_testset.json
```

**Test K Values** (use selected strategy and chunk config):
```bash
for k in 3 5 10; do
    python evaluation/generate_evaluation_results.py \
        --strategies adaptive --top_k $k \
        --index_name haifa-municipality-rag-small-chunks \
        --output_dir "evaluation/comparison_eval_results_per_k/results_k${k}"
done
```

**Evaluate Answer Quality** (LLM as a judge):
```bash
python evaluation/run_llm_judge_evaluation.py \
    --testset_file tests/embedding_testset.json \
    --strategies baseline sentence adaptive --top_k 3 \
    --index_name haifa-municipality-rag-small-chunks
```

**Run Comprehensive Comparison** (all combinations automatically):
```bash
python evaluation/run_comprehensive_comparison.py \
    --testset_file tests/embedding_testset.json
```

### Phase 2 Commands

**Single Config** (uses `run_llm_judge_evaluation.py`):
```bash
python evaluation/run_llm_judge_evaluation.py \
    --testset_file tests/embedding_testset.json \
    --strategies adaptive --top_k 3 \
    --test_enrichment_reranking \
    --output_dir evaluation/llm_judge_eval_results/llm_judge
```

**Multiple Configs** (uses `evaluate_enrichment_reranking_configs.py`):
```bash
python evaluation/evaluate_enrichment_reranking_configs.py \
    --chunk_config small_chunks --k 3 --strategy adaptive \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/enrichment_reranking_results
```
