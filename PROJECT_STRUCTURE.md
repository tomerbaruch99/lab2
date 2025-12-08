# Haifa Municipality RAG Project Structure

## Overview

This project implements a complete RAG (Retrieval-Augmented Generation) system for the Haifa municipality website using:
- **Data Preparation**: Chunks scraped website content
- **Indexing**: Stores chunks in Pinecone vector database
- **Retrieval**: Finds relevant chunks for user questions
- **Prompt Building**: Formats prompts for LLM
- **Generation**: Uses Gemini to generate answers

## Setup Instructions

### Prerequisites

1. **Python 3.9+** installed on your system
   ```bash
   python --version  # Should show 3.9 or higher
   ```

2. **Scraped data file**: `scrape_and_prepare_data/haifa_scraped.json`
   - Download from SharePoint: [haifa_scraped.json](https://technionmail-my.sharepoint.com/:u:/g/personal/amit_shirazi_campus_technion_ac_il/EcLo4Nc_EyBHmCe8jC5R8RsBDPihBFq3K_3LUQRGqRXrNA?e=cbqSSN)
   - Place it in the `scrape_and_prepare_data/` directory
   - Alternatively, use the scraper notebook (`scrape_and_prepare_data/haifa_muni_scraper.ipynb`) to create this file

3. **API keys file**: `utils/api_keys.json`
   - Must be created before running indexing or Gemini examples
   - Format:
   ```json
   {
     "PINECONE_API_KEY": "your-pinecone-api-key-here",
     "GEMINI_API_KEY": "your-gemini-api-key-here"
   }
   ```
   - You can also set `PINECONE_API_KEY` and `GEMINI_API_KEY` as environment variables instead

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Initial Setup (One-Time)

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

## How to Run the Code

### Quick Start: Run the Chatbot

The easiest way to use the system is through the web chatbot:

```bash
streamlit run chatbot.py
```

This opens a web interface where you can ask questions in Hebrew about municipal services.

### Run Individual Components

**Retrieve relevant chunks:**
```bash
python retriever.py \
    --query "איך משלמים ארנונה?" \
    --top_k 5
```

**Get a complete RAG answer:**
```bash
python gemini_integration.py \
    --question "איך משלמים ארנונה?" \
    --top_k 5
```

**Run evaluation:**
```bash
python evaluation/generate_evaluation_results.py \
    --strategies baseline sentence adaptive \
    --top_k 5
```

**Run examples:**
```bash
python examples/example_retriever_usage.py
python examples/example_gemini_rag.py
```

### Common Workflows

**Complete end-to-end workflow:**
1. Prepare data: `python scrape_and_prepare_data/data_preparation.py --input_json scrape_and_prepare_data/haifa_scraped.json --out_dir scrape_and_prepare_data/haifa_prepared_data`
2. Index data: `python indexing.py --prepared_file scrape_and_prepare_data/haifa_prepared_data/haifa_rag_chunks.parquet`
3. Run chatbot: `streamlit run chatbot.py`

**Evaluation workflow:**
1. Generate results: `python evaluation/generate_evaluation_results.py --testset_file tests/embedding_testset.json`
2. Analyze results: `python evaluation/analyze_results.py --results_dir evaluation/evaluation_results`

For more detailed instructions, see the main [README.md](README.md) file.

## File Structure

```
project/
├── scrape_and_prepare_data/
│   ├── data_preparation.py      # Prepares scraped JSON → chunks (Parquet/CSV)
│   ├── haifa_scraped.json       # Input: Scraped website data
│   └── haifa_prepared_data/     # Output: Prepared chunks (created by data_preparation.py)
│       ├── haifa_rag_chunks.parquet
│       └── haifa_rag_chunks.csv
│
├── indexing.py                  # Indexes chunks into Pinecone
├── retriever.py                 # Retrieves relevant chunks from Pinecone
├── prompt_builder.py            # Builds prompts for LLM
├── gemini_integration.py        # Complete RAG system with Gemini
├── confidence_meter.py          # Answer confidence scoring tool
├── chatbot.py                   # Streamlit web UI
├── evaluation/                  # Evaluation scripts and utilities
│   ├── generate_evaluation_results.py     # Generates evaluation results (queries APIs)
│   ├── analyze_evaluation_results.ipynb   # Analyzes results (reads CSV files only)
│   ├── llm_judge.py            # LLM-as-a-judge for answer evaluation
│   └── evaluation_queries.json # Evaluation query set
│
├── examples/                    # Example scripts and tests
│   ├── example_retriever_usage.py   # Examples for retriever
│   ├── example_prompt_builder.py    # Examples for prompt builder
│   ├── example_gemini_rag.py       # Examples for Gemini RAG
│   └── test_gemini_call.py         # Test: Verify Gemini API calls
│
├── utils/                       # Shared utilities
│   ├── __init__.py
│   ├── config.py               # Shared configuration constants
│   ├── pinecone_utils.py       # Pinecone helper functions
│   ├── embedding.py            # Embedding model wrapper
│   ├── query_enhancement.py    # Query rephrasing, enrichment, and reranking
│   ├── smart_page_finder.py   # Tool to return relevant pages to users
│   ├── compare_embedding_models.py  # Compare different embedding models
│   ├── recreate_index.py       # Helper to recreate Pinecone index with correct dimension
│   └── api_keys.json           # API keys (PINECONE_API_KEY, GEMINI_API_KEY)
│
├── run_all_configs.py          # Helper: Run multiple chunk configurations
├── requirements.txt            # Python dependencies
└── README.md                   # Main documentation
```

## Data Flow

```
1. Scraped Data (JSON)
   ↓
2. data_preparation.py
   - Cleans text
   - Chunks content (with overlap)
   - Detects file types
   - Skips generic titles (e.g., "PDF Document")
   - Outputs: Parquet/CSV files
   ↓
3. indexing.py
   - Loads chunks from Parquet/CSV
   - Generates embeddings
   - Stores in Pinecone with metadata
   ↓
4. retriever.py
   - Embeds user question
   - Queries Pinecone
   - Returns top-K chunks
   ↓
5. prompt_builder.py
   - Formats question + chunks
   - Adds system instructions
   - Creates prompt for LLM
   ↓
6. gemini_integration.py
   - Calls Gemini API
   - Returns answer
```

## Key Components

### 1. Data Preparation (`scrape_and_prepare_data/data_preparation.py`)
- **Input**: `haifa_scraped.json` (array of pages with url, title, subtitle, content)
- **Output**: Parquet/CSV files with chunks
- **Features**:
  - Smart chunking (sentence/paragraph boundaries)
  - File type detection (pdf, html, doc, xls, txt)
  - Skips generic titles in embeddings
  - Config-based filenames for comparison

### 2. Indexing (`indexing.py`)
- **Input**: Prepared Parquet file (`haifa_rag_chunks.parquet`)
- **Output**: Pinecone index
- **Features**:
  - Document-based IDs (`doc_id::chunk-{chunk_id}`)
  - Rich metadata (text, chunk_text_only, url, title, subtitle, doc_type, namespace, chunking_strategy, links)
  - Namespace support per chunk
  - Batch processing

### 3. Retrieval (`retriever.py`)
- **Input**: User question
- **Output**: Top-K relevant chunks
- **Features**:
  - Automatic namespace detection from query
  - Strategy filtering (baseline, sentence, adaptive)
  - Metadata filtering (by doc_id, etc.)
  - Fallback to general namespace if no results

### 4. Prompt Building (`prompt_builder.py`)
- **Input**: Question + retrieved chunks
- **Output**: Formatted prompt string
- **Features**:
  - Multiple styles (detailed, concise, conversational, structured)
  - Source citations
  - Conversation history support
  - Custom instructions

### 5. Gemini Integration (`gemini_integration.py`)
- **Input**: Question
- **Output**: Generated answer with confidence score
- **Features**:
  - Complete RAG pipeline
  - Rate limiting with retries
  - Query enhancement (optional): Enriches queries with keywords
  - Reranking (optional): Uses LLM to rerank chunks by relevance
  - Chunking strategy filtering
  - Conversation support
  - Automatic confidence scoring

### 6. Answer Confidence Meter (`confidence_meter.py`)
- **Input**: Retrieved chunks and generated answer
- **Output**: Confidence score and detailed metrics
- **Features**:
  - Calculates average query-chunk similarity (50% weight)
  - Measures retrieval overlap between chunks (30% weight)
  - Detects unsupported claims/hallucinations (20% weight)
  - Returns confidence level (High/Medium/Low) with explanation
  - Visual display in chatbot UI (color-coded meter)

### 7. Query Enhancement (`utils/query_enhancement.py`)
- **Features**:
  - Query rephrasing: Improves question clarity
  - Query enrichment: Adds relevant keywords for better retrieval
  - Chunk reranking: Uses LLM to select most relevant chunks

### 8. LLM Judge (`evaluation/llm_judge.py`)
- **Features**:
  - Evaluates RAG answers against gold standards
  - Multiple quality metrics: correctness, faithfulness, completeness, conciseness, overall

## Default Values

All modules use consistent defaults:

- **Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Pinecone Index**: `haifa-municipality-rag-index`
- **Gemini Model**: `gemini-2.5-flash`
- **API Keys Path**: `utils/api_keys.json`
- **Prepared Data Dir**: `./scrape_and_prepare_data/haifa_prepared_data`
- **Default Config**: `chunk1000_overlap200`

## Chunk Metadata

Each chunk in Pinecone contains:
- `text`: Full text with title/subtitle (for embedding)
- `chunk_text_only`: Just content (for display)
- `doc_id`: Document identifier
- `url`: Source URL
- `title`: Page title
- `subtitle`: Page subtitle
- `doc_type`: Document type (pdf, html, doc, xls, txt)
- `namespace`: Namespace for filtering (arnona, parking, water, etc.)
- `chunking_strategy`: Chunking strategy used (baseline, sentence, adaptive)
- `chunk_id`: Chunk index
- `links`: JSON string containing hyperlinks found in the chunk

## ID Format

Chunks use document-based IDs: `{doc_id}::chunk-{chunk_id}`
- Example: `resident-service::chunk-0`
- Makes it easy to fetch/delete specific documents

## File Type Handling

- **Detection**: Based on URL extension or pattern
- **Filtering**: Can exclude/include specific file types
- **Generic Titles**: PDFs with "PDF Document" title are excluded from embedding context

## Configuration Support

Multiple chunk configurations can be tested:
- `chunk500_overlap100`
- `chunk750_overlap150`
- `chunk1000_overlap200` (default)
- `chunk1500_overlap300`
- `chunk2000_overlap400`

Each configuration creates separate files for comparison.

