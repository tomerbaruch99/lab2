# Haifa Municipality RAG Project Structure

## Overview

This project implements a complete RAG (Retrieval-Augmented Generation) system for the Haifa municipality website using:
- **Data Preparation**: Chunks scraped website content
- **Indexing**: Stores chunks in Pinecone vector database
- **Retrieval**: Finds relevant chunks for user questions
- **Prompt Building**: Formats prompts for LLM
- **Generation**: Uses Gemini to generate answers

## File Structure

```
project/
├── scrape_and_prepare_data/
│   ├── data_preparation.py      # Prepares scraped JSON → chunks (Parquet/CSV)
│   ├── haifa_scraped.json       # Input: Scraped website data
│   └── haifa_prepared_data/     # Output: Prepared chunks (created by data_preparation.py)
│       ├── haifa_paragraph_index_config_chunk1000_overlap200.parquet
│       ├── haifa_paragraph_index_config_chunk1000_overlap200.csv
│       └── haifa_document_index_config_chunk1000_overlap200.parquet
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
- **Input**: Prepared Parquet/CSV files
- **Output**: Pinecone index
- **Features**:
  - Document-based IDs (`doc_id::chunk-{chunk_id}`)
  - Rich metadata (text, chunk_text_only, url, title, subtitle, file_type)
  - Namespace support (dev/prod/language)
  - Batch processing

### 3. Retrieval (`retriever.py`)
- **Input**: User question
- **Output**: Top-K relevant chunks
- **Features**:
  - File type filtering (exclude PDFs, include HTML only, etc.)
  - Metadata filtering (by doc_id, etc.)
  - Batch retrieval
  - Document deletion for reindexing

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
- `file_type`: pdf, html, doc, xls, txt
- `chunk_id`: Chunk index
- `filename`: For compatibility

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

