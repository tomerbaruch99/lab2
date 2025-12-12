# Project Structure

Complete file structure and component overview of the Haifa Municipality RAG project.

## Directory Structure

```
project/
├── Core Components
│   ├── chatbot.py                   # Streamlit web UI
│   ├── gemini_integration.py        # Complete RAG system with Gemini
│   ├── retriever.py                 # Retrieves chunks from Pinecone
│   ├── prompt_builder.py            # Builds prompts for LLM
│   ├── confidence_meter.py          # Answer confidence scoring
│   └── indexing.py                  # Indexes data into Pinecone
│
├── scrape_and_prepare_data/         # Data preparation
│   ├── data_preparation.py          # Prepares scraped JSON → chunks
│   ├── haifa_scraped.json           # Input: Scraped website data
│   ├── haifa_muni_scraper.ipynb     # Scraper notebook
│   ├── haifa_prepared_data/         # Output: Prepared chunks (generated)
│   │   ├── haifa_rag_chunks.parquet
│   │   └── haifa_rag_chunks.csv
│   └── page_index.csv               # Page index for Smart Page Finder (generated)
│
├── evaluation/                      # Evaluation system
│   ├── generate_evaluation_results.py    # Generate CSV results (queries APIs)
│   ├── analyze_evaluation_results.ipynb  # Analyze results (reads CSV only)
│   ├── analyze_results.py               # Command-line analysis
│   ├── evaluate_chunk_configurations.py  # Test multiple chunk configs
│   ├── run_llm_judge_evaluation.py      # LLM judge evaluation
│   ├── llm_judge.py                     # LLM-as-a-judge utility
│   ├── baseline_methods.py              # Baseline methods (TF-IDF, keyword)
│   ├── quantitative_report_generator.py # Report generation
│   ├── evaluation_queries.json          # Evaluation query set
│   ├── chunk_configs.json               # Chunk configuration examples
│   └── README.md                        # Evaluation documentation
│
├── utils/                           # Shared utilities
│   ├── config.py                    # Configuration constants
│   ├── embedding.py                 # Embedding model wrapper
│   ├── pinecone_utils.py            # Pinecone helper functions
│   ├── query_enhancement.py         # Query enrichment and reranking
│   ├── smart_page_finder.py         # Smart Page Finder tool
│   ├── build_page_index.py          # Build page index
│   ├── api_keys.json                # API keys (create this file)
│   └── README.md                    # Utilities documentation
│
├── examples/                        # Usage examples
│   ├── example_retriever_usage.py   # Basic retrieval examples
│   ├── example_prompt_builder.py    # Prompt building examples
│   ├── example_gemini_rag.py        # Complete RAG pipeline
│   ├── example_smart_page_finder.py # Smart Page Finder usage
│   ├── example_retrieval_diagnostics.py # Retrieval diagnostics
│   ├── example_chatbot_integration.py   # Chatbot integration
│   ├── test_gemini_call.py          # Simple Gemini API test
│   └── README.md                    # Examples documentation
│
├── tests/                           # Test data
│   └── embedding_testset.json       # Test set with ground truth labels
│
├── logos/                           # Logo and image assets
│
├── Configuration Files
│   ├── requirements.txt             # Python dependencies
│   ├── pyvenv.cfg                   # Virtual environment config
│   └── PROJECT_STRUCTURE.md         # This file
│
└── Documentation
    ├── README.md                    # Main documentation
    └── evaluation/EVALUATION_WORKFLOW.md
```

## Component Overview

### Core RAG Pipeline

1. **Data Preparation** (`scrape_and_prepare_data/data_preparation.py`)
   - Input: `haifa_scraped.json`
   - Output: Chunked data (Parquet/CSV)
   - Features: Smart chunking, file type detection, namespace assignment

2. **Indexing** (`indexing.py`)
   - Input: Prepared chunks
   - Output: Pinecone index
   - Features: Embedding generation, metadata storage, batch processing

3. **Retrieval** (`retriever.py`)
   - Input: User query
   - Output: Top-K relevant chunks
   - Features: Namespace detection, strategy filtering, fallback search

4. **Prompt Building** (`prompt_builder.py`)
   - Input: Query + chunks
   - Output: Formatted prompt
   - Features: Multiple styles, source citations, conversation history

5. **Generation** (`gemini_integration.py`)
   - Input: Query
   - Output: Generated answer
   - Features: Complete RAG pipeline, query enhancement, reranking, confidence scoring

### Supporting Components

- **`chatbot.py`** - Streamlit web interface with Hebrew support
- **`confidence_meter.py`** - Answer quality evaluation (🟢/🟡/🔴)
- **`utils/smart_page_finder.py`** - Suggests relevant official pages
- **`evaluation/`** - Comprehensive evaluation system for comparing strategies

## Data Flow

```
Scraped JSON
    ↓
data_preparation.py → Chunks (Parquet/CSV)
    ↓
indexing.py → Pinecone Index
    ↓
retriever.py → Relevant Chunks
    ↓
prompt_builder.py → Formatted Prompt
    ↓
gemini_integration.py → Answer + Confidence
```

## Key Metadata Fields

Each chunk in Pinecone contains:
- `text` - Full text with title/subtitle (for embedding)
- `chunk_text_only` - Content only (for display)
- `doc_id` - Document identifier
- `url` - Source URL
- `title`, `subtitle` - Page metadata
- `doc_type` - File type (pdf, html, doc, xls, txt)
- `namespace` - Category (arnona, parking, water, etc.)
- `chunking_strategy` - Strategy used (baseline, sentence, adaptive)
- `chunk_id` - Chunk index
- `links` - Hyperlinks found in chunk

## Default Configuration

- **Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Pinecone Index**: `haifa-municipality-rag-index`
- **Gemini Model**: `gemini-3-pro-preview` (RAG generation), `gemini-2.5-pro` (LLM judge)
- **API Keys Path**: `utils/api_keys.json`
- **Chunk Config**: 1000 chars, 200 overlap (default)

## ID Format

Chunks use document-based IDs: `{doc_id}::chunk-{chunk_id}`

Example: `resident-service::chunk-0`

This format makes it easy to fetch or delete specific documents.
