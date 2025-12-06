# Chunking Strategy Evaluation Guide

This guide explains how to use the automated evaluation script to compare chunking strategies and generate evaluation reports for your project.

## Overview

The `evaluate_chunking_strategies.py` script evaluates the performance of different chunking strategies:
- **baseline**: Simple character-based chunking
- **sentence**: Sentence-aware chunking
- **adaptive**: Dynamic strategy selection based on document type

## Prerequisites

1. **Indexed Data**: Your data must be indexed in Pinecone using `indexing_2.py`
2. **API Keys**: `utils/api_keys.json` with `PINECONE_API_KEY`
3. **Dependencies**: Install required packages (pandas, matplotlib, seaborn, etc.)

## Quick Start

### Basic Usage

```bash
python evaluate_chunking_strategies.py \
    --output_dir ./evaluation_results
```

This uses default evaluation queries and tests all 3 strategies.

### Using Custom Queries

```bash
python evaluate_chunking_strategies.py \
    --queries_file evaluation_queries.json \
    --output_dir ./evaluation_results
```

### Customizing Strategies

```bash
python evaluate_chunking_strategies.py \
    --strategies baseline sentence adaptive \
    --top_k 5 \
    --output_dir ./evaluation_results
```

## Query File Format

Create a JSON file with your evaluation queries:

```json
[
  {
    "query": "איך משלמים ארנונה?",
    "expected_namespace": "arnona",
    "category": "payment"
  },
  {
    "query": "מה המחיר של חניה?",
    "expected_namespace": "parking",
    "category": "pricing"
  }
]
```

**Fields:**
- `query`: The Hebrew question to evaluate
- `expected_namespace`: Expected namespace (for accuracy measurement)
- `category`: Query category (for analysis grouping)

## Output Files

The evaluation generates several files in the output directory:

### 1. **evaluation_results.csv**
Raw results with one row per query-strategy combination:
- Query text
- Strategy used
- Retrieval scores (avg, max, min, std)
- Namespace detection accuracy
- Number of results
- Document type distribution

### 2. **strategy_statistics.csv**
Aggregate statistics per strategy:
- Average scores
- Score distributions
- Namespace accuracy
- Document diversity

### 3. **namespace_statistics.csv**
Namespace detection accuracy by namespace and strategy.

### 4. **evaluation_report.md**
Comprehensive markdown report with:
- Summary statistics
- Strategy comparison tables
- Best strategy recommendations
- Performance by category

### 5. **Visualizations**
- `strategy_comparison.png`: Bar charts comparing strategies
- `namespace_accuracy_heatmap.png`: Heatmap of namespace detection accuracy
- `category_analysis.png`: Performance by query category

## Metrics Explained

### Retrieval Quality Metrics

1. **Average Score**: Mean similarity score of retrieved chunks
   - Higher = Better retrieval quality
   - Range: 0-1 (typically 0.3-0.9)

2. **Score Distribution**: Spread of scores
   - Lower std = More consistent retrieval
   - Higher std = Variable quality

3. **Unique Documents**: Number of different documents retrieved
   - Higher = More diverse results
   - Lower = More focused on fewer documents

### Namespace Accuracy

- **Detection Accuracy**: Percentage of queries where namespace was correctly detected
- Measured against expected namespace from query file
- Important for routing efficiency

### Document Type Distribution

- Shows which document types (pdf, event, procedural, etc.) are retrieved
- Helps understand retrieval patterns

## Interpreting Results

### Best Strategy Selection

1. **Check Average Score**: Highest avg_score indicates best retrieval quality
2. **Check Consistency**: Lower std_score means more reliable
3. **Check Namespace Accuracy**: Higher accuracy means better routing
4. **Check Document Diversity**: Balance between diversity and relevance

### Example Interpretation

```
Strategy: adaptive
- Average Score: 0.7523 (highest)
- Namespace Accuracy: 0.85 (85%)
- Recommendation: Use adaptive for best overall performance
```

### Category-Specific Performance

The script identifies best strategy per query category:
- **Procedural queries**: Often work best with sentence chunking
- **Event queries**: May work best with paragraph/adaptive chunking
- **Payment queries**: Varies by content structure

## Advanced Usage

### Evaluating Single Strategy

```bash
python evaluate_chunking_strategies.py \
    --strategies adaptive \
    --output_dir ./eval_adaptive_only
```

### Custom Top-K

```bash
python evaluate_chunking_strategies.py \
    --top_k 10 \
    --output_dir ./evaluation_results
```

### Using Different API Keys

```bash
python evaluate_chunking_strategies.py \
    --api_keys_path /path/to/api_keys.json \
    --output_dir ./evaluation_results
```

## Integration with Project Report

The evaluation results can be directly used in your project report:

1. **Strategy Comparison Section**:
   - Use `strategy_comparison.png` figure
   - Reference `strategy_statistics.csv` for numbers

2. **Namespace Analysis Section**:
   - Use `namespace_accuracy_heatmap.png` figure
   - Reference `namespace_statistics.csv` for accuracy metrics

3. **Category Analysis Section**:
   - Use `category_analysis.png` figure
   - Show best strategy per category

4. **Recommendations Section**:
   - Use findings from `evaluation_report.md`
   - Quote best performing strategy

## Troubleshooting

### No Results Returned

- Check that data is indexed in Pinecone
- Verify index name matches `retriever_2.py` default
- Check namespace exists in index

### Low Scores

- May indicate embedding model mismatch
- Check if queries are in Hebrew (required)
- Verify data quality in index

### Namespace Detection Issues

- Review namespace detection rules in `retriever_2.py`
- Add missing keywords to namespace rules
- Check query format (Hebrew required)

## Example Workflow

```bash
# 1. Prepare data (if not already done)
python scrape_and_prepare_data/data_preparation_2.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --out_dir scrape_and_prepare_data/haifa_prepared_data

# 2. Index data (if not already done)
python indexing_2.py \
    --prepared_file scrape_and_prepare_data/haifa_prepared_data/haifa_rag_chunks.parquet

# 3. Run evaluation
python evaluate_chunking_strategies.py \
    --queries_file evaluation_queries.json \
    --output_dir ./evaluation_results

# 4. Review results
cat evaluation_results/evaluation_report.md
open evaluation_results/strategy_comparison.png
```

## Next Steps

After evaluation:

1. **Select Best Strategy**: Based on metrics, choose optimal strategy
2. **Refine Queries**: Add more diverse queries to evaluation set
3. **Namespace Optimization**: Update namespace detection rules if needed
4. **Report Writing**: Use results in evaluation section of project report

## Support

For issues or questions:
- Check `ANALYSIS_chunking_changes.md` for background on changes
- Review `retriever_2.py` for retrieval implementation details
- Check evaluation results CSV files for detailed metrics

