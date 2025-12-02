# Haifa Municipality RAG Project

This project implements a RAG (Retrieval-Augmented Generation) system for the Haifa municipality website, allowing users to ask questions about municipal services, regulations, and information.

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.9+** installed on your system
   ```bash
   python --version  # Should show 3.9 or higher
   ```

2. **Scraped data file**: `scrape_and_prepare_data/haifa_scraped.json`
   - This file must exist before running data preparation
   - The scraper notebook (`scrape_and_prepare_data/haifa_muni_scraper.ipynb`) creates this file

3. **API keys file**: `utils/api_keys.json`
   - **Must be created before running indexing or Gemini examples**
   - Format:
   ```json
   {
     "PINECONE_API_KEY": "your-pinecone-api-key-here",
     "GEMINI_API_KEY": "your-gemini-api-key-here"
   }
   ```
   - You can also set `PINECONE_API_KEY` and `GEMINI_API_KEY` as environment variables instead

4. **Dependencies**: Install required packages
   ```bash
   pip install -r requirements.txt
   ```

## Quickstart (End-to-End)

Here's a single, linear path to get the system running:

### 1. Prepare data

```bash
python scrape_and_prepare_data/data_preparation.py
```

This processes `scrape_and_prepare_data/haifa_scraped.json` and creates prepared chunks in `haifa_prepared_data/`.

### 2. Index into Pinecone

```bash
python indexing.py
```

This indexes the prepared chunks into Pinecone using embeddings.

### 3. Ask a question via Gemini RAG

```bash
python gemini_integration.py \
    --question "איך משלמים ארנונה?" \
    --top_k 5 \
    --exclude_file_types pdf
```

This retrieves relevant chunks and generates an answer using Gemini.

That's it! You now have a working RAG system. For more details on each step, see the sections below.

## Project Structure

```
project/
├── scrape_and_prepare_data/
│   ├── data_preparation.py    # Prepares scraped data for RAG indexing
│   ├── haifa_scraped.json    # Scraped data from Haifa municipality website
│   └── haifa_muni_scraper.ipynb
├── indexing.py               # Indexes prepared data into Pinecone
├── retriever.py             # Retrieves relevant chunks from Pinecone
├── prompt_builder.py        # Builds prompts for LLM
├── gemini_integration.py    # Complete RAG system with Gemini
├── examples/                # Example scripts and tests
│   ├── example_retriever_usage.py
│   ├── example_prompt_builder.py
│   ├── example_gemini_rag.py
│   └── test_gemini_call.py
├── utils/                   # Shared utilities
│   ├── config.py
│   ├── pinecone_utils.py
│   └── embedding.py
├── run_all_configs.py       # Helper script for multiple configurations
├── haifa_prepared_data/      # Output directory (created by data_preparation.py)
├── utils/
│   └── api_keys.json        # API keys (PINECONE_API_KEY, GEMINI_API_KEY)
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Data Preparation

The data preparation process converts the scraped JSON data into a format suitable for RAG indexing.

### Input Format

The input JSON file (`haifa_scraped.json`) should contain an array of page objects, each with:
- `url`: The page URL
- `title`: Page title
- `subtitle`: Page subtitle (optional)
- `content`: The main content text

### Usage

```bash
cd scrape_and_prepare_data
python data_preparation.py \
    --input_json haifa_scraped.json \
    --out_dir ../haifa_prepared_data \
    --chunk_chars 1000 \
    --chunk_overlap 200
```

Or from the project root:
```bash
python scrape_and_prepare_data/data_preparation.py \
    --input_json scrape_and_prepare_data/haifa_scraped.json \
    --out_dir haifa_prepared_data \
    --chunk_chars 1000 \
    --chunk_overlap 200
```

### Parameters

- `--input_json`: Path to the input JSON file (default: `./scrape_and_prepare_data/haifa_scraped.json`)
- `--out_dir`: Output directory for prepared data (default: `./scrape_and_prepare_data/haifa_prepared_data`)
- `--chunk_chars`: Maximum characters per chunk (default: 1000)
- `--chunk_overlap`: Character overlap between chunks (default: 200)
- `--config_suffix`: Optional config suffix for filenames (default: auto-generated from chunk_chars and chunk_overlap)
- `--run_all_configs`: Run multiple chunk size/overlap configurations

### Output

The script generates (with config suffix, e.g., `chunk1000_overlap200`):
- `haifa_paragraph_index_config_{config_suffix}.parquet`: Main chunk index in Parquet format
- `haifa_paragraph_index_config_{config_suffix}.csv`: Same data in CSV format (fallback)
- `haifa_document_index_config_{config_suffix}.parquet`: Document-level summary

Each chunk contains:
- `doc_id`: Document identifier (derived from URL)
- `url`: Original page URL
- `filename`: Filename for compatibility
- `chunk_id`: Chunk index within document
- `start_char`: Character position in original text
- `text`: Full chunk text (includes title/subtitle context)
- `title`: Page title
- `subtitle`: Page subtitle
- `chunk_text_only`: Just the chunk content without title/subtitle

## Indexing

After data preparation, index the chunks into Pinecone for retrieval.

> **Note**: Make sure you've completed the prerequisites above, especially creating `utils/api_keys.json` with your `PINECONE_API_KEY`.

### Usage

```bash
python indexing.py \
    --prepared_dir ./scrape_and_prepare_data/haifa_prepared_data \
    --api_keys_path utils/api_keys.json \
    --embedding_model all-MiniLM-L6-v2 \
    --index_name haifa-municipality-rag-index \
    --batch_size 128
```

### Parameters

- `--prepared_dir`: Directory with prepared data (default: `./scrape_and_prepare_data/haifa_prepared_data`)
- `--paragraph_parquet`: Parquet filename (default: `haifa_paragraph_index_config_chunk1000_overlap200.parquet`)
- `--paragraph_csv`: CSV filename fallback (default: `haifa_paragraph_index_config_chunk1000_overlap200.csv`)
- `--config`: Config suffix to use (e.g., `chunk1000_overlap200`). If provided, overrides paragraph_parquet/csv.
- `--api_keys_path`: Path to API keys file (default: `utils/api_keys.json`)
- `--embedding_model`: SentenceTransformer model name (default: `all-MiniLM-L6-v2`)
- `--index_name`: Pinecone index name (default: `haifa-municipality-rag-index`)
- `--batch_size`: Batch size for embedding/upsert (default: 128)
- `--namespace`: Optional namespace for dev/prod/language separation

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

For a complete end-to-end setup, see the [Quickstart](#quickstart-end-to-end) section above.

## Chunking Strategy

The data preparation script uses a smart chunking strategy:
- Chunks are created with a maximum character limit
- Overlap between chunks ensures context preservation
- Chunk boundaries prefer sentence endings (periods, exclamation marks, question marks)
- If no sentence boundary is found, paragraph boundaries are used
- Each chunk includes title and subtitle context for better retrieval

## Retrieval

After indexing, you can retrieve relevant chunks using the retriever module.

### Usage as Script

```bash
python retriever.py \
    --query "איך משלמים ארנונה?" \
    --top_k 5 \
    --index_name haifa-municipality-rag-index
```

### Usage as Module

```python
from retriever import Retriever

# Initialize retriever
retriever = Retriever(
    api_keys_path="utils/api_keys.json",
    embedding_model_name="all-MiniLM-L6-v2",
    index_name="haifa-municipality-rag-index",
    namespace="dev"  # Optional: for dev/prod/language separation
)

# Retrieve top-K results
results = retriever.retrieve("איך משלמים ארנונה?", top_k=5)

# Access results
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Title: {result.get('title', 'N/A')}")
    print(f"URL: {result.get('url', 'N/A')}")
    print(f"Content: {result.get('chunk_text_only', '')}")
    print()
```

### Advanced Features

**Filter by document:**
```python
filter_dict = {"doc_id": "resident-service"}
results = retriever.retrieve(query, top_k=5, filter_dict=filter_dict)
```

**Batch retrieval:**
```python
queries = ["איך משלמים ארנונה?", "מוקדי שירות"]
all_results = retriever.retrieve_batch(queries, top_k=3)
```

**Delete document for reindexing:**
```python
retriever.delete_by_doc_id("resident-service")
```

### Retriever Parameters

- `--query`: User question/query string (required)
- `--top_k`: Number of top results to return (default: 5)
- `--index_name`: Pinecone index name (default: haifa-municipality-rag-index)
- `--namespace`: Optional namespace (e.g., 'dev', 'prod', 'hebrew', 'arabic')
- `--filter_doc_id`: Filter results by specific doc_id
- `--show_full_text`: Show full text (with title/subtitle) instead of chunk_text_only

## Notes

- The content is in Hebrew, so ensure your embedding model supports Hebrew text
- The default model `all-MiniLM-L6-v2` works reasonably well for multilingual content
- For better Hebrew support, consider using a multilingual model like `paraphrase-multilingual-MiniLM-L12-v2`
- The retriever uses the same embedding model as indexing for consistency
- Document-based IDs (e.g., `"doc_id::chunk-0"`) make it easy to fetch/delete specific documents

## Prompt Building

After retrieving relevant chunks, format them into a prompt for the language model.

### Usage

```python
from retriever import Retriever
from prompt_builder import PromptBuilder

# Retrieve chunks
retriever = Retriever(...)
chunks = retriever.retrieve("איך משלמים ארנונה?", top_k=5)

# Build prompt
builder = PromptBuilder()
prompt = builder.build_prompt("איך משלמים ארנונה?", chunks)

# Send to LLM (e.g., Gemini, GPT)
# response = llm.generate(prompt)
```

### Prompt Styles

The prompt builder supports different styles:

- **DETAILED** (default): Full context with sources, titles, URLs
- **CONCISE**: Minimal formatting, compact
- **CONVERSATIONAL**: Natural, friendly format
- **STRUCTURED**: Highly structured with clear sections

```python
from prompt_builder import PromptBuilder, PromptStyle

builder = PromptBuilder(style=PromptStyle.CONCISE)
prompt = builder.build_prompt(question, chunks)
```

### Features

- **Source citations**: Automatically includes URLs and titles
- **Conversation history**: Support for multi-turn conversations
- **Custom instructions**: Override default system instructions
- **Chunk formatting**: Truncates long chunks, formats metadata
- **Hebrew support**: Properly handles Hebrew text and RTL

### Example Output

The prompt builder creates prompts like:

```
אתה עוזר AI מומחה של עיריית חיפה...

============================================================
מידע רלוונטי מאתר עיריית חיפה:
============================================================

[מקור 1]
כותרת: מוקדי השירות
קישור: https://www.haifa.muni.il/service-center/
רלוונטיות: 0.852

מוקד עירוני 106
טלפון: 04-8356356

============================================================

שאלת המשתמש:
איך משלמים ארנונה?

בבקשה ענה על השאלה על בסיס המידע שסופק לעיל.
```

## Gemini Integration

Complete RAG system with Gemini for generating answers.

> **Note**: Make sure you've completed the prerequisites above. Your `utils/api_keys.json` must include both `PINECONE_API_KEY` and `GEMINI_API_KEY` before running Gemini examples.

### Usage

**Simple usage:**
```python
from gemini_integration import GeminiRAG

# Initialize RAG system
rag = GeminiRAG(api_keys_path="utils/api_keys.json")

# Ask a question
result = rag.answer_question("איך משלמים ארנונה?", top_k=5)
print(result["answer"])
```

**Exclude PDFs (get HTML/TXT only):**
```python
result = rag.answer_question(
    "מה מספר הטלפון של המוקד?",
    top_k=5,
    exclude_file_types=["pdf"]
)
```

**With conversation history:**
```python
history = [
    {"role": "user", "content": "מה שעות הפעילות?"},
    {"role": "assistant", "content": "המוקד פעיל 24 שעות."},
]

result = rag.answer_with_conversation(
    "ואיך משלמים ארנונה?",
    history,
    top_k=5
)
```

**CLI usage:**
```bash
python gemini_integration.py \
    --question "איך משלמים ארנונה?" \
    --top_k 5 \
    --exclude_file_types pdf
```

### Features

- **Automatic retrieval**: Retrieves relevant chunks from Pinecone
- **Prompt building**: Formats prompts with context and sources
- **Gemini integration**: Uses Google Gemini 2.5 Flash for generation
- **Rate limiting**: Handles API rate limits with exponential backoff
- **Error handling**: Robust error handling and retries
- **File type filtering**: Exclude/include specific file types
- **Conversation support**: Multi-turn conversations with history

### API Compatibility

The integration uses `google.generativeai` and follows Gemini's expected format:
- Plain text prompts (string format)
- Direct `generate_content()` calls
- Proper response text extraction
- Rate limit handling with exponential backoff

## Web UI Integration

To integrate this RAG system into a web chatbot or application, you can call `GeminiRAG.answer_question()` from your web backend. Here's a simple example:

```python
from flask import Flask, request, jsonify
from gemini_integration import GeminiRAG

app = Flask(__name__)
rag = GeminiRAG(api_keys_path="api_keys.json", log_file_path="logs/interactions.jsonl")

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "")
    result = rag.answer_question(question, top_k=5)
    return jsonify({"answer": result["answer"]})

if __name__ == "__main__":
    app.run()
```

The `GeminiRAG` class handles all the complexity (retrieval, prompt building, generation) and returns a simple dictionary with the answer. You can integrate it into any web framework (Flask, FastAPI, Django, etc.) or deploy it as a REST API service.

### Logging

Enable interaction logging for traceability by passing `log_file_path` to `GeminiRAG`:
```python
rag = GeminiRAG(
    api_keys_path="utils/api_keys.json",
    log_file_path="logs/interactions.jsonl"  # Logs all questions and answers
)
```

Logs are written in JSONL format and include the question, retrieved chunks (with doc_id, url, score), and whether an answer was generated or "no relevant info" was returned.

