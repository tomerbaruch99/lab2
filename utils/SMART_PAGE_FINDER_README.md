# Smart Page Finder - URL Recommender

A lightweight tool that suggests relevant pages from the Haifa municipality website based on user queries using semantic similarity.

## Location

The Smart Page Finder is located in `utils/` for easy access by the chatbot application:

- **Main module**: `utils/smart_page_finder.py` - The SmartPageFinder class
- **Build script**: `utils/build_page_index.py` - Script to build the page index from scraped data
- **Page index**: `scrape_and_prepare_data/page_index.csv` - Pre-computed embeddings (created by build script)

## Quick Start

### 1. Build the Page Index (one-time setup)

After scraping the website, build the page index:

```python
from utils.build_page_index import build_page_index

# Uses default paths:
# - Input: scrape_and_prepare_data/haifa_scraped_with_hiperlinks.json
# - Output: scrape_and_prepare_data/page_index.csv
build_page_index()
```

Or from command line:

```bash
python -m utils.build_page_index
```

### 2. Use in Chatbot

```python
from utils.smart_page_finder import SmartPageFinder

# Initialize (loads page index automatically)
finder = SmartPageFinder()

# Find relevant pages for a query
query = "איך אני מחדש תו חניה?"
results = finder.find_relevant_pages(query, top_k=5)

# Format results
print(finder.format_results(results, include_scores=True))
```

### 3. Integration with Chatbot

See `examples/example_chatbot_integration.py` for a complete example of how to integrate Smart Page Finder with your chatbot to suggest relevant official pages alongside RAG responses.

## Features

- **Semantic similarity**: Uses embeddings to match user queries to page titles
- **Pre-indexed**: All 6,000+ pages are pre-embedded for fast lookup
- **Lightweight**: Simple CSV-based index, no database required
- **Integrated**: Uses the same embedding model as your RAG system (`paraphrase-multilingual-MiniLM-L12-v2`)

## API

### SmartPageFinder Class

```python
finder = SmartPageFinder(
    page_index_path=None,  # Auto-detects: scrape_and_prepare_data/page_index.csv
    embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
    top_k=5
)

# Find relevant pages
results = finder.find_relevant_pages(query, top_k=5)

# Format results
formatted = finder.format_results(results, include_scores=False)
```

### Results Format

Each result is a dictionary with:
- `title`: Page title
- `subtitle`: Page subtitle (if available)
- `url`: Page URL
- `score`: Cosine similarity score (0-1)

## Notes

- The page index must be built before using SmartPageFinder
- The embedding model must match the one used to build the index
- The tool uses CPU by default to avoid CUDA compatibility issues

