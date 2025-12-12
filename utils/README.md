# Utilities

Shared utility modules for the RAG system.

## Core Modules

- **`config.py`** - Shared configuration constants
  - Default embedding model, index name, API keys path
  - Namespace rules, chunking parameters

- **`embedding.py`** - Embedding model wrapper
  - `EmbeddingModel` class for generating embeddings
  - Supports multiple SentenceTransformer models

- **`pinecone_utils.py`** - Pinecone helper functions
  - API key loading
  - Index management utilities

## Advanced Features

- **`query_enhancement.py`** - Query enhancement and reranking
  - Query rephrasing and enrichment
  - Reranking retrieved chunks by relevance

- **`smart_page_finder.py`** - Smart Page Finder tool
  - Finds relevant pages based on user queries
  - See `SMART_PAGE_FINDER_README.md` for details

## Helper Scripts

- **`build_page_index.py`** - Builds page index for Smart Page Finder
  - Creates `scrape_and_prepare_data/page_index.csv`

## Configuration

- **`api_keys.json`** - API keys file (create this file)
  - Must contain `PINECONE_API_KEY` and `GEMINI_API_KEY`
  - See main README for format

## Usage Examples

For detailed usage examples of these utilities, see:
- `examples/example_retriever_usage.py` - Using retriever and embedding models
- `examples/example_prompt_builder.py` - Prompt building examples
- `examples/example_smart_page_finder.py` - Smart Page Finder usage
- `examples/example_gemini_rag.py` - Complete RAG pipeline with query enhancement

See `examples/README.md` for a complete list of examples.
