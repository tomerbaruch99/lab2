# Smart Page Finder

Lightweight tool that suggests relevant pages from the Haifa municipality website based on user queries using semantic similarity.

## Quick Start

### Setup

1. **Build page index** (one-time):
   ```bash
   python utils/build_page_index.py
   ```
   Creates `scrape_and_prepare_data/page_index.csv`

2. **Use in code**:
   ```python
   from utils.smart_page_finder import SmartPageFinder
   
   finder = SmartPageFinder()
   pages = finder.find_relevant_pages("איך משלמים ארנונה?", top_k=5)
   ```

## Features

- Semantic similarity matching using embeddings
- Pre-indexed for fast lookup (6,000+ pages)
- Uses same embedding model as RAG system
- Lightweight CSV-based index

## API

```python
finder = SmartPageFinder(
    page_index_path=None,  # Auto-detects: scrape_and_prepare_data/page_index.csv
    embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# Find relevant pages
results = finder.find_relevant_pages(query, top_k=5)

# Format results
formatted = finder.format_results(results, include_scores=False)
```

**Result format**: Each result contains `title`, `subtitle`, `url`, `score`.

## Requirements

- `scrape_and_prepare_data/haifa_scraped.json` must exist
- Page index must be built before use
- Embedding model must match the one used to build the index

## Integration

See `utils/smart_page_finder.py` for the implementation and `chatbot.py` for usage examples.
