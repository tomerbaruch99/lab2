# Evaluation Folder

This folder contains all files related to evaluating chunking strategies for the RAG system.

## Files

### Results Generation (Requires API Access)
- **`generate_evaluation_results.py`** - Generates evaluation results
  - Queries Pinecone and Gemini APIs
  - Saves all results to CSV files locally
  - Requires API keys and internet connection
  - Does NOT generate visualizations

- **`evaluate_chunk_configurations.py`** - Evaluates different chunk size/overlap configurations
  - Automates the workflow: prepare → index → evaluate
  - Tests multiple configurations in one run
  - Requires API access for indexing and evaluation

### Analysis (No API Access Required)
- **`analyze_results.py`** - Command-line analysis script
  - Loads locally stored CSV files only
  - Generates visualizations and statistical analysis
  - Does NOT query any APIs
  - Can run offline after results are generated

- **`analyze_evaluation_results.ipynb`** - Interactive Jupyter notebook for analysis
  - Loads CSV results from disk
  - Generates visualizations inline for easy viewing
  - Provides detailed analysis and comparisons
  - Does NOT require API access

### Data
- **`evaluation_queries.json`** - Set of Hebrew queries for evaluation
  - Contains 20 queries covering different namespaces and categories
  - Can be customized for your evaluation needs

- **`../tests/embedding_testset.json`** - Testset with ground truth labels (recommended)
  - Contains queries with labeled relevant/irrelevant documents
  - Enables precision, recall, and accuracy metrics
  - Automatically used if available when no testset file is specified

- **`chunk_configs_example.json`** - Example configuration file for testing different chunk sizes/overlaps

### Utilities
- **`baseline_methods.py`** - Baseline retrieval methods for comparison (TF-IDF, keyword matching)
- **`llm_judge.py`** - LLM-as-a-judge utility for evaluating answer quality

## Quick Start

### Two-Phase Workflow

The evaluation is organized into two separate phases:

1. **Phase 1: Generate Results** (requires API access)
   - Queries Pinecone and Gemini
   - Saves results to CSV files
   
2. **Phase 2: Analyze Results** (no API access needed)
   - Reads locally stored CSV files
   - Generates visualizations and analysis
   - Can run offline

### Phase 1: Generate Results (Python Script)

Run the Python script to generate CSV results:

**Option A: Using testset with ground truth (Recommended)**
```bash
cd evaluation
python generate_evaluation_results.py \
    --testset_file ../tests/embedding_testset.json \
    --output_dir ./evaluation_results
```

This enables precision, recall, and accuracy metrics based on ground truth labels.

**Option B: Using custom queries**
```bash
cd evaluation
python generate_evaluation_results.py \
    --queries_file evaluation_queries.json \
    --output_dir ./evaluation_results
```

**Option C: Default (auto-detects testset if available)**
```bash
cd evaluation
python generate_evaluation_results.py \
    --output_dir ./evaluation_results
```

If `../tests/embedding_testset.json` exists, it will be used automatically. Otherwise, uses default queries.

**Output**: CSV files saved to `evaluation/evaluation_results/`:
- `evaluation_results.csv` - Raw evaluation data
- `strategy_statistics.csv` - Aggregate statistics
- `score_distribution.csv` - Query distribution across score categories
- Additional files for baseline comparisons if `--include_baselines` is used

### Phase 2: Analyze Results (No API Access Required)

After generating results, analyze them using either method:

#### Option A: Command-Line Analysis
```bash
python evaluation/analyze_results.py \
    --results_dir evaluation/evaluation_results
```

Generates visualizations and prints summary statistics.

#### Option B: Jupyter Notebook Analysis
1. Open `evaluation/analyze_evaluation_results.ipynb` in Jupyter
2. Run all cells to load CSV results and generate visualizations
3. View inline plots and analysis directly in the notebook

**Note**: Both analysis methods only read local CSV files and do NOT require API access.

## Output Files

### From Python Script

CSV files are saved to the output directory (default: `evaluation_results/`):
- `evaluation_results.csv` - Raw evaluation data (one row per query-strategy combination)
- `strategy_statistics.csv` - Aggregate statistics per strategy
- `namespace_statistics.csv` - Namespace detection accuracy

### From Notebook

Visualizations and analysis:
- Strategy comparison charts
- Namespace accuracy heatmaps
- Category analysis plots
- Performance distributions

## Metrics Explained

### Retrieval Quality Metrics

1. **Average Score**: Mean similarity score of retrieved chunks (0-1, typically 0.3-0.9)
   - Higher = Better retrieval quality

2. **Score Distribution**: Spread of scores (std)
   - Lower std = More consistent retrieval
   - Higher std = Variable quality

3. **Unique Documents**: Number of different documents retrieved
   - Higher = More diverse results
   - Lower = More focused on fewer documents

### Ground Truth Metrics (when using testset)

When using `tests/embedding_testset.json` or a testset with ground truth labels:

1. **Precision**: Fraction of retrieved chunks that are relevant (0-1)
2. **Recall**: Fraction of relevant documents that were retrieved (0-1)
3. **Accuracy**: Overall correctness of retrieval (0-1)

These metrics provide objective evaluation compared to similarity scores alone.

### Namespace Accuracy

- **Detection Accuracy**: Percentage of queries where namespace was correctly detected
- Measured against expected namespace from query file
- Important for routing efficiency

## Chunking Strategies

The evaluation tests three chunking strategies:

- **baseline**: Simple character-based chunking
- **sentence**: Sentence-aware chunking
- **adaptive**: Dynamic strategy selection based on document type

## Baseline Methods (Optional)

To include baseline methods for comparison, use `--include_baselines`:

```bash
python generate_evaluation_results.py \
    --include_baselines \
    --testset_file ../tests/embedding_testset.json
```

This adds three baseline "strategies" to the evaluation:
- `baseline_tfidf`: TF-IDF keyword-based retrieval (requires scikit-learn)
- `baseline_keyword`: Simple keyword matching
- `baseline_retrieval_only`: Semantic search without LLM generation

## Evaluating Different Chunk Sizes and Overlaps

Chunk sizes and overlaps are set during data preparation, not during evaluation. To test different configurations:

### Quick Method (Manual)

**Step 1: Prepare data with specific chunk config**
```bash
python scrape_and_prepare_data/data_preparation.py \
    --input_json scrape_and_prepare_data/haifa_scraped_data.json \
    --out_dir ./prepared_data/chunk_500_overlap_100 \
    --chunk_chars 500 \
    --chunk_overlap 100
```

**Step 2: Index the data**
```bash
python indexing.py \
    --prepared_file ./prepared_data/chunk_500_overlap_100/haifa_rag_chunks.parquet \
    --index_name haifa-municipality-rag-chunk500 \
    --api_keys_path utils/api_keys.json
```

**Step 3: Run evaluation** (specify the index name)
```bash
python generate_evaluation_results.py \
    --testset_file ../tests/embedding_testset.json \
    --output_dir ./evaluation_results/chunk_500_overlap_100 \
    --index_name haifa-municipality-rag-chunk500
```

Alternatively, use an environment variable:
```bash
PINECONE_INDEX_NAME=haifa-municipality-rag-chunk500 \
python generate_evaluation_results.py \
    --testset_file ../tests/embedding_testset.json \
    --output_dir ./evaluation_results/chunk_500_overlap_100
```

### Automated Method (Helper Script)

Use the `evaluate_chunk_configurations.py` helper script:

**Step 1: Create a config file** (see `chunk_configs_example.json`):
```json
[
    {
        "name": "small_chunks",
        "chunk_chars": 500,
        "chunk_overlap": 100
    },
    {
        "name": "medium_chunks",
        "chunk_chars": 1000,
        "chunk_overlap": 200
    }
]
```

**Step 2: Run the evaluation script**
```bash
python evaluation/evaluate_chunk_configurations.py \
    --input_json scrape_and_prepare_data/haifa_scraped_data.json \
    --configs evaluation/chunk_configs_example.json \
    --output_base_dir ./chunk_config_evaluations \
    --testset_file tests/embedding_testset.json
```

This automatically:
1. Prepares data for each configuration
2. Indexes each to a separate Pinecone index
3. Runs evaluation for each
4. Generates comparison results

### Recommended Chunk Configurations to Test

- **Small chunks** (500 chars, 100 overlap): Better for precise retrieval
- **Medium chunks** (1000 chars, 200 overlap): Balanced (default)
- **Large chunks** (2000 chars, 400 overlap): Better for context preservation
- **High overlap** (1000 chars, 400 overlap): Better for boundary handling
- **Low overlap** (1000 chars, 50 overlap): Faster indexing, less redundancy

## Query File Formats

The evaluation supports two formats:

### Format 1: Simple Queries (evaluation_queries.json)

```json
[
    {
        "query": "איך משלמים ארנונה?",
        "expected_namespace": "arnona",
        "category": "payment"
    }
]
```

**Fields:**
- `query`: The Hebrew question to evaluate
- `expected_namespace`: Expected namespace (for accuracy measurement)
- `category`: Query category (for analysis grouping)

### Format 2: Testset with Ground Truth (tests/embedding_testset.json)

```json
{
    "queries": [
        {
            "query": "איך משלמים ארנונה?",
            "documents": [
                {
                    "text": "עיריית חיפה מאפשרת לשלם ארנונה...",
                    "label": "relevant"
                },
                {
                    "text": "מידע לא רלוונטי...",
                    "label": "irrelevant"
                }
            ]
        }
    ]
}
```

**Fields:**
- `query`: The Hebrew question to evaluate
- `documents`: List of documents with `text` and `label` ("relevant" or "irrelevant")
- This format enables precision, recall, and accuracy metrics

**Recommendation**: Use the testset format for proper evaluation with ground truth labels.

## Command Line Options

```bash
python generate_evaluation_results.py \
    --queries_file evaluation_queries.json \      # Custom queries file
    --testset_file ../tests/embedding_testset.json \  # Testset with ground truth
    --output_dir ./evaluation_results \          # Output directory
    --strategies baseline sentence adaptive \     # Strategies to test
    --top_k 5 \                                  # Number of chunks to retrieve
    --index_name haifa-municipality-rag \        # Pinecone index name
    --api_keys_path utils/api_keys.json \       # API keys file
    --include_baselines                           # Include baseline methods
```

## Requirements

Make sure you have:
- Data indexed in Pinecone (run `indexing.py` first)
- `../utils/api_keys.json` with `PINECONE_API_KEY`
- Required Python packages:
  ```bash
  pip install pandas numpy matplotlib seaborn tqdm
  ```
- For baseline methods (optional):
  ```bash
  pip install scikit-learn
  ```

## LLM-as-a-Judge for Answer Evaluation

The project includes an LLM judge utility (`evaluation/llm_judge.py`) for evaluating RAG answer quality:

```python
from evaluation.llm_judge import judge_answer
from gemini_integration import init_gemini, load_api_keys

api_keys = load_api_keys("utils/api_keys.json")
gemini_model = init_gemini(api_keys, "gemini-2.5-flash")

scores = judge_answer(
    question="איך משלמים ארנונה?",
    gold_answer="תשובה נכונה מהמסמך...",
    rag_answer="תשובה מה-RAG...",
    gemini_model=gemini_model
)
```

The judge evaluates answers on 5 metrics (0.0-1.0):
- **correctness**: Factual accuracy
- **faithfulness**: Adherence to source material (no hallucination)
- **completeness**: Coverage of important information
- **conciseness**: Clarity and brevity
- **overall**: General quality score

## Troubleshooting

### No Results Returned
- Check that data is indexed in Pinecone
- Verify index name matches (use `--index_name` or environment variable)
- Check namespace exists in index

### Low Scores
- May indicate embedding model mismatch
- Check if queries are in Hebrew (required)
- Verify data quality in index

### Namespace Detection Issues
- Review namespace detection rules in `retriever.py`
- Add missing keywords to namespace rules
- Check query format (Hebrew required)

## Example Workflow

```bash
# 1. Prepare data (if not already done)
python scrape_and_prepare_data/data_preparation.py \
    --input_json scrape_and_prepare_data/haifa_scraped_data.json \
    --out_dir scrape_and_prepare_data/haifa_prepared_data

# 2. Index data (if not already done)
python indexing.py \
    --prepared_file scrape_and_prepare_data/haifa_prepared_data/haifa_rag_chunks.parquet

# 3. Run evaluation
python evaluation/generate_evaluation_results.py \
    --testset_file tests/embedding_testset.json \
    --output_dir ./evaluation_results

# 4. Visualize results
jupyter notebook evaluation/analyze_evaluation_results.ipynb
```

## Notes

- The notebook version is recommended for interactive analysis and viewing plots
- The script version is better for automated runs or batch processing
- All plots in the notebook are displayed inline for convenience
- **Chunk sizes and overlaps are set during data preparation**, not during evaluation
- To test different configurations, you must re-prepare and re-index the data
