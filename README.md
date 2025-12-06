# Haifa Municipality RAG Project

This project implements a RAG (Retrieval-Augmented Generation) system for the Haifa municipality website, allowing users to ask questions about municipal services, regulations, and information.

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.9+** installed on your system
   ```bash
   python --version  # Should show 3.9 or higher
   ```

2. **Scraped data file**: `scrape_and_prepare_data/haifa_scraped_with_hiperlinks.json`
   - **Download from SharePoint**: [haifa_scraped_with_hiperlinks.json](https://technionmail-my.sharepoint.com/:u:/g/personal/amit_shirazi_campus_technion_ac_il/ETm0fnq5sGpAtznQTJ7zpZoB0rvYPnz4r1VG42VcX1suJA?e=LHHFSK)
   - Download the file and place it in the `scrape_and_prepare_data/` directory
   - This file must exist before running data preparation
   - Alternatively, you can use the scraper notebook (`scrape_and_prepare_data/haifa_muni_scraper.ipynb`) to create this file

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

## Usage Guide

This project supports three main use cases. Choose the workflow that matches your goal:

| Use Case | Goal | Quick Command | Section |
|---------|------|---------------|---------|
| **🚀 Run Application** | Use chatbot to ask questions | `streamlit run chatbot.py` | [Running the Application](#running-the-application) |
| **📊 Evaluate Project** | Compare strategies & analyze performance | `cd evaluation && python generate_evaluation_results.py` | [Evaluating the Project](#evaluating-the-project) |
| **💡 See Examples** | Learn how components work | `python examples/example_retriever_usage.py` | [Running Examples](#running-examples) |

### 🚀 I Want to Run the Application
**Use case**: Use the chatbot to ask questions about Haifa municipality services.

**Quick start:**
1. Set up data (one-time): `python scrape_and_prepare_data/data_preparation.py`
2. Index data (one-time): `python indexing.py`
3. Run chatbot: `streamlit run chatbot.py`

**See**: [Running the Application](#running-the-application) section below for detailed steps.

---

### 📊 I Want to Evaluate the Project
**Use case**: Compare chunking strategies, analyze performance, and generate evaluation reports.

**Quick start:**
1. Generate results: `cd evaluation && python generate_evaluation_results.py`
2. Analyze results: `jupyter notebook analyze_evaluation_results.ipynb` or `python analyze_results.py`

**See**: [Evaluating the Project](#evaluating-the-project) section below for detailed steps.

---

### 💡 I Want to See Examples
**Use case**: Learn how individual components work with small examples.

**Quick start:**
```bash
python examples/example_retriever_usage.py
python examples/example_gemini_rag.py
```

**See**: [Running Examples](#running-examples) section below for all available examples.

---

## Running the Application

**Goal**: Use the chatbot to ask questions about Haifa municipality services.

### Prerequisites

Before running the application, you need to set up the data:

1. **Prepare data** (one-time setup):
   ```bash
   python scrape_and_prepare_data/data_preparation.py
   ```
   This processes the scraped data and creates prepared chunks.

2. **Index into Pinecone** (one-time setup):
   ```bash
   python indexing.py \
       --prepared_file scrape_and_prepare_data/haifa_prepared_data/haifa_rag_chunks.parquet
   ```
   This indexes the prepared chunks into Pinecone using embeddings.

### Running the Chatbot

**Option 1: Web UI (Recommended)**

```bash
streamlit run chatbot.py
```

This opens a web interface in your browser where you can:
- Ask questions in Hebrew about municipal services
- Get answers powered by Gemini RAG
- See relevant page suggestions from Smart Page Finder
- View chat history and export conversations

**Option 2: Command Line**

```bash
python gemini_integration.py \
    --question "איך משלמים ארנונה?" \
    --top_k 5
```

**With advanced features:**
```bash
# With query enhancement (improves retrieval)
python gemini_integration.py \
    --question "איך משלמים ארנונה?" \
    --top_k 5 \
    --use_query_enhancement

# With reranking (improves chunk selection)
python gemini_integration.py \
    --question "איך משלמים ארנונה?" \
    --top_k 5 \
    --use_reranking
```

This runs a single question through the RAG system and prints the answer.

### What You'll See

- **Web UI**: Interactive chat interface with Hebrew (RTL) support
- **Command Line**: Text output with the answer and optionally retrieved chunks

---

## Evaluating the Project

**Goal**: Compare chunking strategies, analyze retrieval performance, and generate evaluation reports.

### Prerequisites

- Data must be indexed in Pinecone (see [Running the Application](#running-the-application))
- Evaluation requires `utils/api_keys.json` with `PINECONE_API_KEY`

### Step 1: Run Evaluation

Run the evaluation script to test different chunking strategies:

```bash
python evaluation/generate_evaluation_results.py \
    --queries_file evaluation/evaluation_queries.json \
    --strategies baseline sentence adaptive \
    --top_k 5
```

**With baseline comparisons:**
```bash
python evaluation/generate_evaluation_results.py \
    --queries_file evaluation/evaluation_queries.json \
    --include_baselines
```

*Note: Results are saved to `evaluation/evaluation_results/` by default.*

This script:
- Tests queries across all specified strategies
- Optionally includes baseline methods (TF-IDF, keyword matching, retrieval-only) for comparison
- Saves results to CSV files (no visualizations)
- Takes a few minutes depending on number of queries and baselines included

**Note:** Baseline methods help contextualize system performance by comparing against:
- Traditional keyword-based search (TF-IDF)
- Simple exact matching approaches
- Retrieval without generation (to show LLM contribution)

**Output**: CSV files in `evaluation/evaluation_results/`:
- `evaluation_results.csv` - Raw results per query-strategy
- `strategy_statistics.csv` - Aggregate statistics per strategy
- `namespace_statistics.csv` - Namespace detection accuracy

### Step 2: Analyze Results (No API Access Required)

After generating results, analyze them using either method:

**Option A: Command-Line Analysis**
```bash
python evaluation/analyze_results.py \
    --results_dir evaluation/evaluation_results
```

**Option B: Jupyter Notebook**
```bash
jupyter notebook evaluation/analyze_evaluation_results.ipynb
```

Both analysis methods:
- ✅ Load CSV files from Step 1
- ✅ Generate visualizations (strategy comparison, heatmaps, category analysis)
- ✅ Provide summary statistics and recommendations
- ❌ Do NOT query Pinecone or Gemini
- ❌ Do NOT require API keys
- ✅ Can run offline

**Output**: Visualization plots saved to `evaluation/evaluation_results/`:
- `strategy_comparison.png` - Bar charts comparing strategies
- `namespace_accuracy_heatmap.png` - Namespace detection accuracy
- `category_analysis.png` - Performance by query category

### Customizing Evaluation

**Use custom queries:**
```bash
python evaluation/generate_evaluation_results.py \
    --queries_file my_custom_queries.json \
    --output_dir ./custom_results
```

**Test specific strategies:**
```bash
python evaluation/generate_evaluation_results.py \
    --strategies adaptive \
    --output_dir ./adaptive_only
```

**Adjust retrieval parameters:**
```bash
python evaluation/generate_evaluation_results.py \
    --top_k 10
```

For detailed evaluation documentation, see `evaluation/README.md`.

---

## Running Examples

**Goal**: Learn how individual components work with small, focused examples.

### Available Examples

The `examples/` directory contains scripts demonstrating specific features:

#### 1. Basic Retrieval
```bash
python examples/example_retriever_usage.py
```
Shows how to:
- Initialize the retriever
- Query Pinecone for relevant chunks
- Filter by chunking strategy
- Understand namespace detection

#### 2. Prompt Building
```bash
python examples/example_prompt_builder.py
```
Demonstrates:
- Different prompt styles (detailed, concise, conversational, structured)
- Building prompts with retrieved chunks
- Custom system instructions

#### 3. Complete RAG Pipeline
```bash
python examples/example_gemini_rag.py
```
Shows:
- Full RAG workflow (retrieval → prompt building → generation)
- Using different chunking strategies
- Conversation history support
- Custom instructions

#### 4. Smart Page Finder
```bash
python examples/example_smart_page_finder.py
```
Demonstrates:
- Finding relevant pages based on user queries
- Formatting page recommendations
- Integrating with chatbot responses

#### 5. Retrieval Diagnostics
```bash
python examples/example_retrieval_diagnostics.py
```
Helps diagnose:
- Why retrieval might return poor results
- File type distribution in results
- Namespace detection accuracy
- Strategy performance differences

#### 6. Test Gemini API
```bash
python examples/test_gemini_call.py
```
Simple test to verify:
- Gemini API connection works
- API keys are configured correctly
- Basic prompt generation

### Running Examples

Most examples can be run directly:

```bash
# From project root
python examples/example_retriever_usage.py
```

Some examples may require:
- API keys in `utils/api_keys.json`
- Data indexed in Pinecone
- Specific configuration

Check the comments in each example file for requirements.

### Example Output

Examples typically print:
- Step-by-step execution
- Retrieved chunks with scores
- Generated prompts
- Final answers or recommendations

Use examples to understand how to integrate components into your own code.

---

## Initial Setup (One-Time)

Before using any of the workflows above, complete the initial setup:

## Project Structure

```
project/
├── scrape_and_prepare_data/
│   ├── haifa_prepared_data/   # Output directory (created by data_preparation.py)
│   ├── data_preparation.py    # Prepares scraped data for RAG indexing
│   ├── haifa_scraped_with_hiperlinks.json    # Scraped data from Haifa municipality website (download from SharePoint - see Prerequisites)
│   └── haifa_muni_scraper.ipynb
├── indexing.py               # Indexes prepared data into Pinecone
├── retriever.py             # Retrieves relevant chunks from Pinecone
├── prompt_builder.py        # Builds prompts for LLM (includes EVAL style)
├── gemini_integration.py    # Complete RAG system with Gemini
├── confidence_meter.py      # Answer confidence scoring tool
├── chatbot.py               # Streamlit web UI integrated with RAG, Smart Page Finder, and Confidence Meter
├── evaluation/              # Evaluation scripts and reports
│   ├── generate_evaluation_results.py     # Generates evaluation results (queries APIs)
│   ├── analyze_evaluation_results.ipynb   # Analyzes results (reads CSV files only)
│   ├── llm_judge.py        # LLM-as-a-judge for answer evaluation
│   ├── evaluation_queries.json          # Evaluation query set
│   └── README.md                        # Evaluation guide
├── examples/                # Example scripts and tests
│   ├── example_retriever_usage.py
│   ├── example_prompt_builder.py
│   ├── example_gemini_rag.py
│   ├── example_chatbot_integration.py
│   ├── example_smart_page_finder.py
│   ├── example_retrieval_diagnostics.py
│   └── test_gemini_call.py
├── utils/                   # Shared utilities
│   ├── config.py            # Shared configuration constants
│   ├── pinecone_utils.py    # Pinecone helper functions
│   ├── embedding.py         # Embedding model wrapper
│   ├── query_enhancement.py # Query rephrasing, enrichment, and reranking
│   ├── smart_page_finder.py # Tool to return relevant pages to users
│   ├── build_page_index.py  # Builds page index for Smart Page Finder
│   └── api_keys.json        # API keys (PINECONE_API_KEY, GEMINI_API_KEY)
├── run_all_configs.py       # Helper script for multiple configurations
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Data Preparation

The data preparation process converts the scraped JSON data into a format suitable for RAG indexing.

### Input Format

The input JSON file (`haifa_scraped_with_hiperlinks.json`) should contain an array of page objects, each with:
- `url`: The page URL
- `title`: Page title
- `subtitle`: Page subtitle (optional)
- `content`: The main content text

### Usage

```bash
cd scrape_and_prepare_data
python data_preparation.py \
    --input_json haifa_scraped_with_hiperlinks.json \
    --out_dir ./haifa_prepared_data \
    --chunk_chars 1000 \
    --chunk_overlap 200
```

Or from the project root:
```bash
python scrape_and_prepare_data/data_preparation.py \
    --input_json scrape_and_prepare_data/haifa_scraped_with_hiperlinks.json \
    --out_dir haifa_prepared_data \
    --chunk_chars 1000 \
    --chunk_overlap 200
```

### Parameters

- `--input_json`: Path to the input JSON file (default: `./scrape_and_prepare_data/haifa_scraped_with_hiperlinks.json`)
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
    --embedding_model paraphrase-multilingual-MiniLM-L12-v2 \
    --index_name haifa-municipality-rag-index \
    --batch_size 128
```

### Parameters

- `--prepared_dir`: Directory with prepared data (default: `./scrape_and_prepare_data/haifa_prepared_data`)
- `--paragraph_parquet`: Parquet filename (default: `haifa_paragraph_index_config_chunk1000_overlap200.parquet`)
- `--paragraph_csv`: CSV filename fallback (default: `haifa_paragraph_index_config_chunk1000_overlap200.csv`)
- `--config`: Config suffix to use (e.g., `chunk1000_overlap200`). If provided, overrides paragraph_parquet/csv.
- `--api_keys_path`: Path to API keys file (default: `utils/api_keys.json`)
- `--embedding_model`: SentenceTransformer model name (default: `paraphrase-multilingual-MiniLM-L12-v2`)
- `--index_name`: Pinecone index name (default: `haifa-municipality-rag-index`)
- `--batch_size`: Batch size for embedding/upsert (default: 128)
- `--namespace`: Optional namespace for dev/prod/language separation

### Troubleshooting: Dimension Mismatch Error

If you get an error like `Vector dimension 768 does not match the dimension of the index 384`, it means your existing Pinecone index was created with a different embedding model. You need to recreate the index with the correct dimension.

**Option 1: Use the helper script (Recommended)**
```bash
python utils/recreate_index.py --index_name haifa-municipality-rag-index
```

**Option 2: Manually delete and recreate**
```python
from pinecone import Pinecone
from utils import load_pinecone_api_key, DEFAULT_API_KEYS_PATH

# Load API key and initialize
api_key = load_pinecone_api_key(DEFAULT_API_KEYS_PATH)
pc = Pinecone(api_key=api_key)

# Delete old index
pc.delete_index("haifa-municipality-rag-index")

# Wait for deletion to complete, then run indexing.py
# It will automatically create a new index with the correct dimension
```

**Important**: Changing embedding models requires re-indexing all your data, as embeddings are model-specific.

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
    embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
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
- The default model `paraphrase-multilingual-MiniLM-L12-v2` is optimized for Hebrew and multilingual content
- This model was selected based on performance comparison for Hebrew text similarity
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
- **Chunking strategy filtering**: Filter by chunking strategy (baseline, sentence, adaptive)
- **Query enhancement** (optional): Enriches queries with additional keywords for better retrieval
- **Reranking** (optional): Uses LLM to rerank retrieved chunks by relevance
- **Conversation support**: Multi-turn conversations with history
- **Smart Page Finder integration**: Automatically suggests relevant pages

### API Compatibility

The integration uses `google.generativeai` and follows Gemini's expected format:
- Plain text prompts (string format)
- Direct `generate_content()` calls
- Proper response text extraction
- Rate limit handling with exponential backoff

## Answer Confidence Meter

The project includes an **Answer Confidence Meter** tool that evaluates the quality and reliability of RAG-generated answers. This tool provides visual feedback to help users understand how confident the system is in its responses.

### What It Does

After generating an answer, the system calculates a confidence score based on three factors:

1. **Average similarity between query and retrieved chunks** (50% weight)
   - Uses the retrieval similarity scores from Pinecone
   - Higher scores indicate better query-chunk matching

2. **Retrieval overlap score** (30% weight)
   - Measures how much the retrieved chunks agree with each other
   - Uses semantic similarity between chunk pairs
   - Higher overlap indicates consistent information across chunks

3. **Supported claim ratio** (20% weight)
   - Detects potential hallucinations by checking if answer claims are supported by retrieved chunks
   - Uses semantic similarity to verify answer sentences against chunk content
   - Higher ratio means fewer unsupported claims

### Confidence Score Calculation

```python
confidence_score = (
    0.5 * avg_chunk_similarity +
    0.3 * retrieval_overlap_score +
    0.2 * supported_claim_ratio
) * 100
```

### Confidence Levels

- **High** (≥70%): Green indicator 🟢
  - High similarity between query and chunks
  - Chunks strongly agree with each other
  - No unsupported claims detected

- **Medium** (40-69%): Yellow indicator 🟡
  - Moderate similarity or some disagreement
  - Most claims appear supported

- **Low** (<40%): Red indicator 🔴
  - Low similarity or significant disagreement
  - Some claims may not be fully supported

### Usage

The confidence meter is automatically integrated into the chatbot UI and appears below each answer. It can also be accessed programmatically:

```python
from gemini_integration import GeminiRAG
from confidence_meter import calculate_confidence

rag = GeminiRAG(api_keys_path="utils/api_keys.json")
result = rag.answer_question("איך משלמים ארנונה?", top_k=5)

# Confidence information is included in the result
confidence = result.get("confidence", {})
print(f"Confidence: {confidence['confidence_score']}% ({confidence['confidence_level']})")
print(f"Reason: {confidence['reason']}")
```

### Output Format

The confidence meter returns a dictionary with:

```python
{
    "confidence_score": 82.5,  # Overall score (0-100)
    "confidence_level": "High",  # "High", "Medium", or "Low"
    "avg_chunk_similarity": 85.2,  # Query-chunk similarity (0-100)
    "retrieval_overlap": 78.3,  # Chunk agreement (0-100)
    "supported_claim_ratio": 90.1,  # Supported claims (0-100)
    "reason": "High similarity between user query and retrieved chunks; retrieved chunks strongly agree with each other; no unsupported claims detected.",
    "unsupported_claims": []  # List of potentially unsupported claims
}
```

### Visual Display

In the chatbot UI, the confidence meter displays:
- Color-coded indicator (🟢/🟡/🔴) based on confidence level
- Progress bar showing confidence percentage
- Human-readable explanation of the confidence level
- RTL-compatible styling for Hebrew interface

### Why It's Valuable

- **Evaluates RAG pipeline**: Helps identify when retrieval or generation quality is poor
- **Builds user trust**: Users can see when answers are highly confident vs. uncertain
- **Great for presentations**: Visual green/yellow/red meter is impressive and easy to understand
- **Debugging tool**: Low confidence scores can indicate issues with retrieval or chunk quality

## Web Chatbot

The project includes a fully functional Streamlit chatbot (`chatbot.py`) that integrates the RAG system with Smart Page Finder and the Answer Confidence Meter.

### Running the Chatbot

```bash
streamlit run chatbot.py
```

The chatbot provides:
- **RAG-powered answers**: Uses Gemini RAG to answer questions based on retrieved chunks
- **Answer Confidence Meter**: Visual confidence indicator for each answer (🟢/🟡/🔴)
- **Smart Page Finder**: Automatically suggests relevant official pages from the Haifa municipality website
- **Hebrew (RTL) support**: Full right-to-left text support with Gisha font
- **Chat history**: Maintains conversation history during the session
- **Export functionality**: Export chat history as text file

### Features

- Real-time question answering using the RAG system
- Automatic confidence scoring for every answer
- Visual confidence meter with color-coded indicators
- Automatic page recommendations based on query similarity
- Clean, user-friendly Hebrew interface
- Session-based chat history

## Smart Page Finder

The Smart Page Finder (`utils/smart_page_finder.py`) is a tool that returns relevant pages to users based on their queries. It uses semantic similarity to find the most relevant official pages from the Haifa municipality website.

### Usage

```python
from utils.smart_page_finder import SmartPageFinder

finder = SmartPageFinder()
pages = finder.find_relevant_pages("איך משלמים ארנונה?", top_k=3)

for page in pages:
    print(f"{page['title']}: {page['url']}")
```

### Building the Page Index

Before using Smart Page Finder, you need to build the page index:

```bash
python utils/build_page_index.py
```

This creates `scrape_and_prepare_data/page_index.csv` with page embeddings.

### Evaluation System

The project includes a comprehensive evaluation system for comparing chunking strategies and baseline methods.

**Strategies compared:**
- **baseline**: Simple character-based chunking
- **sentence**: Sentence-aware chunking
- **adaptive**: Dynamic strategy selection based on document type

**Baseline methods** (for comparison and benchmarking):
- **TF-IDF Baseline**: Traditional keyword-based retrieval using TF-IDF vectorization
- **Keyword Matching Baseline**: Simple exact keyword matching (Jaccard similarity)
- **Retrieval-Only Baseline**: Semantic search without LLM generation

**Metrics:**
- Retrieval quality (average similarity scores with statistical analysis)
- **Score distribution**: Explicit proportions of queries in each performance category
- **Baseline comparison**: Quantitative improvement metrics (mean, median, proportion better)
- **Statistical significance**: Paired t-tests comparing strategies vs baselines (p-values)
- Namespace detection accuracy
- Document diversity
- Performance by query category

**Quantitative Analysis:**
The evaluation provides comprehensive quantitative analysis including:
- Score distributions showing what **proportion of queries** achieve each performance level
- Improvement percentages over baselines with statistical significance tests
- Explicit quantitative context to determine if scores are genuinely strong vs. merely adequate

**Usage:** See [Evaluating the Project](#evaluating-the-project) section above.

**Include baselines in evaluation:**
```bash
python evaluation/generate_evaluation_results.py \
    --include_baselines
```

**Detailed documentation:** See `evaluation/README.md`

## Web UI Integration (Custom Backend)

To integrate this RAG system into a custom web application, you can call `GeminiRAG.answer_question()` from your web backend. Here's a simple example:

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

