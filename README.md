# Haifa Municipality RAG Project

RAG (Retrieval-Augmented Generation) system for the Haifa municipality website, allowing users to ask questions about municipal services, regulations, and information.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download scraped data (see Prerequisites)
# Place haifa_scraped.json in scrape_and_prepare_data/

# 3. Create API keys file: utils/api_keys.json with PINECONE_API_KEY and GEMINI_API_KEY

# 4. Prepare and index data
python scrape_and_prepare_data/data_preparation.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --out_dir scrape_and_prepare_data/haifa_prepared_data

python indexing.py \
    --prepared_file scrape_and_prepare_data/haifa_prepared_data/haifa_rag_chunks.parquet

# 5. Run the chatbot
streamlit run chatbot.py
```

## Prerequisites

1. **Python 3.9+**
2. **Scraped data**: Download [haifa_scraped.json](https://technionmail-my.sharepoint.com/:u:/g/personal/amit_shirazi_campus_technion_ac_il/EcLo4Nc_EyBHmCe8jC5R8RsBDPihBFq3K_3LUQRGqRXrNA?e=cbqSSN) and place in `scrape_and_prepare_data/`
3. **API keys**: Create `utils/api_keys.json`:
   ```json
   {
     "PINECONE_API_KEY": "your-key",
     "GEMINI_API_KEY": "your-key"
   }
   ```
4. **Dependencies**: `pip install -r requirements.txt`
   - Optional: `pip install scikit-learn>=1.0.0` for baseline evaluation methods

## Usage

### Run Application
```bash
streamlit run chatbot.py
```
Web UI with Hebrew support, confidence meter, and Smart Page Finder. See [Running the Application](#running-the-application) for details.

### Evaluate Project
```bash
python evaluation/generate_evaluation_results.py
python evaluation/analyze_results.py
```
Compare chunking strategies and analyze performance. See `evaluation/README.md` for details.

### Examples
See the `examples/` folder for detailed usage examples:
- `example_retriever_usage.py` - Basic retrieval examples
- `example_prompt_builder.py` - Prompt building examples
- `example_gemini_rag.py` - Complete RAG pipeline
- `example_smart_page_finder.py` - Smart Page Finder usage
- `example_retrieval_diagnostics.py` - Retrieval diagnostics
- `example_chatbot_integration.py` - Chatbot integration

See `examples/README.md` for detailed descriptions and usage instructions.

### Learn More
See the main components (`retriever.py`, `gemini_integration.py`, `chatbot.py`) for usage examples.

## Running the Application

### Setup (One-Time)

1. **Prepare data**:
   ```bash
   python scrape_and_prepare_data/data_preparation.py \
       --input_json scrape_and_prepare_data/haifa_scraped.json \
       --out_dir scrape_and_prepare_data/haifa_prepared_data
   ```

2. **Index into Pinecone**:
   ```bash
   python indexing.py \
       --prepared_file scrape_and_prepare_data/haifa_prepared_data/haifa_rag_chunks.parquet
   ```

3. **Build page index** (optional, for Smart Page Finder):
   ```bash
   python utils/build_page_index.py
   ```

### Run Chatbot

**Web UI**:
```bash
streamlit run chatbot.py
```

**Command Line**:
```bash
python gemini_integration.py --question "איך משלמים ארנונה?" --top_k 5
```

**With enhancements**:
```bash
python gemini_integration.py \
    --question "איך משלמים ארנונה?" \
    --top_k 5 \
    --use_query_enhancement \
    --use_reranking
```

## Evaluating the Project

**Quick start**:
```bash
# Generate results (requires API access)
python evaluation/generate_evaluation_results.py \
    --testset_file tests/embedding_testset.json \
    --output_dir evaluation/evaluation_results

# Analyze results (no API access needed)
python evaluation/analyze_results.py \
    --results_dir evaluation/evaluation_results
```

See `evaluation/README.md` for detailed documentation.

## Project Structure

```
project/
├── scrape_and_prepare_data/  # Data preparation (see README.md)
├── indexing.py               # Index data into Pinecone
├── retriever.py             # Retrieve chunks from Pinecone
├── prompt_builder.py        # Build prompts for LLM
├── gemini_integration.py    # Complete RAG system
├── confidence_meter.py      # Answer confidence scoring
├── chatbot.py               # Streamlit web UI
├── examples/                # Usage examples (see README.md)
├── evaluation/              # Evaluation scripts (see README.md)
└── utils/                   # Utilities (see README.md)
```

## Key Components

- **Retriever**: Automatic namespace detection, metadata-aware retrieval, fallback search
- **Prompt Builder**: Multiple styles (detailed, concise, conversational, structured)
- **Gemini Integration**: Complete RAG pipeline with query enhancement and reranking
- **Confidence Meter**: Evaluates answer quality (🟢/🟡/🔴)
- **Smart Page Finder**: Suggests relevant official pages
- **Evaluation System**: Compare strategies, baseline methods, and configurations

## Documentation

- **Examples**: See `examples/README.md` for usage examples and code samples
- **Data Preparation**: See `scrape_and_prepare_data/README.md`
- **Evaluation**: See `evaluation/README.md`
- **Utilities**: See `utils/README.md`
- **Smart Page Finder**: See `utils/SMART_PAGE_FINDER_README.md`

## Troubleshooting

**Dimension mismatch error**: Delete and recreate the Pinecone index with the correct embedding model dimension.

**No results returned**: Check that data is indexed and index name matches.

**Low scores**: Verify embedding model matches, queries are in Hebrew, and data quality.
