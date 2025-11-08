# Contract Clause Q&A Assistant

A Retrieval-Augmented Generation (RAG) system for answering questions about contract clauses using the CUAD (Contract Understanding Atticus Dataset) dataset. The system combines dense and sparse retrieval methods with reranking, query understanding, and metadata filtering to provide accurate answers to legal questions.

## Features

- **Data Preparation Pipeline**: Processes CUAD dataset into structured format with train/val/test splits
- **Dual-Index, Dual-Stage Retrieval**: 
  - **Dense retrieval**: Legal-BERT embeddings with FAISS for semantic similarity
  - **Sparse retrieval**: BM25 for keyword-based matching (legal terms, phrases)
  - **Reranking**: Cross-encoder (BGE reranker) for final ranking
- **Query Understanding**: Zero-shot classification to map free-form questions to CUAD categories
- **Metadata Filtering**: Filter by filename (for single-contract queries) or category (for focused searches)
- **Question Answering**: GPT-4o-mini powered Q&A with citation support
- **Embedding Generation**: Support for multiple embedding models (Jina, Legal-BERT, MPNet)
- **Caching**: BM25 index caching for faster repeated queries

## Project Structure

```
.
├── app.py                      # Main CLI application
├── qa_assistant.py            # Question answering interface
├── rag_retriever.py           # Hybrid retrieval system (dense + sparse + rerank + filters)
├── query_understanding.py     # Zero-shot category classification
├── embeddings.py              # Embedding generation utilities
├── vectorstore.py             # FAISS-based vector store
├── bm25_index.py              # BM25 sparse retrieval with metadata filtering
├── reranker.py                # BGE reranker integration
├── example_query.py           # Interactive retrieval demo
├── data_preparation.py        # CUAD data processing pipeline
├── consts.py                  # Configuration constants
├── data/                      # CUAD dataset files
│   └── CUAD_v1/
├── cuad_prepared_data/        # Processed data outputs
└── processed_data/            # Additional processed outputs (embeddings, BM25 cache)
```

## Setup

### Prerequisites

- Python 3.8+
- CUAD dataset (download from [Atticus Project](https://www.atticusproject.org/))

### Installation

1. Install required dependencies:
```bash
pip install pandas numpy sentence-transformers faiss-cpu openai rank-bm25 FlagEmbedding
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. Prepare the data:
   - Place CUAD dataset in `data/CUAD_v1/`
   - Run data preparation pipeline (see `data_preperation_README.md` for details)

4. Generate embeddings:
```bash
python embeddings.py
```

5. Build BM25 index (optional, will be built automatically on first use):
```bash
python -c "from bm25_index import build_bm25; build_bm25()"
```

## Usage

### Interactive Retrieval Demo

Run the example query interface to test retrieval:
```bash
python example_query.py
```

Example questions:
- "Can the contract be terminated without cause?" → Maps to "Termination For Convenience"
- "What are the non-compete restrictions?" → Maps to "Non-Compete"
- "Is there a governing law clause?" → Maps to "Governing Law"

### Interactive Q&A

Run the main application:
```bash
python app.py
```

### Programmatic Usage

#### Basic Retrieval

```python
from rag_retriever import retrieve

# Simple retrieval with query understanding
results = retrieve("Can the contract be terminated without cause?")

# Each result contains:
# - text: The clause text
# - filename: Source contract filename
# - category: CUAD category
# - answer: Answer text (if available)
# - question_template: Template question for this category
```

#### Retrieval with Metadata Filtering

```python
from rag_retriever import retrieve

# Filter by filename (for single-contract queries)
results = retrieve(
    "What are the termination terms?",
    filename="contract_123.pdf"
)

# Filter by category (for focused searches)
results = retrieve(
    "Can they end this without cause?",
    category="Termination For Convenience"
)

# Disable query understanding and use explicit category
results = retrieve(
    "termination clause",
    category="Termination For Convenience",
    use_query_understanding=False
)
```

#### Query Understanding

```python
from query_understanding import map_query_to_category, get_category_candidates

# Map query to single category
category, confidence = map_query_to_category("Can they end this without cause?")
# Returns: ("Termination For Convenience", 0.85)

# Get top-k category candidates
candidates = get_category_candidates("non-compete restrictions", top_k=3)
# Returns: [("Non-Compete", 0.92), ("Competitive Restriction Exception", 0.45), ...]
```

#### Advanced Retrieval Parameters

```python
from rag_retriever import retrieve

results = retrieve(
    query="What are the liability limits?",
    k_dense=10,        # Number of dense retrieval results
    k_bm25=10,         # Number of BM25 retrieval results
    final_k=5,         # Final number of results after reranking
    filename=None,     # Optional: filter by filename
    category=None,     # Optional: filter by category (or auto-detected)
    use_query_understanding=True,  # Enable automatic category detection
    confidence_threshold=0.3        # Minimum confidence for category mapping
)
```

## Components

### Data Preparation (`data_preparation.py`)

Processes the CUAD master CSV into a long-format table with:
- Clause text cleaning and normalization
- Contract type inference
- Train/val/test splits at contract level
- Optional chunking and SQuAD JSON flattening

See `data_preperation_README.md` for detailed documentation.

### Embeddings (`embeddings.py`)

Generates embeddings for contract clauses using various models:
- **Jina Embeddings v2**: `jinaai/jina-embeddings-v2-base-en`
- **Legal-BERT**: `nlpaueb/legal-bert-base-uncased`
- **MPNet**: `sentence-transformers/all-mpnet-base-v2`

### Retrieval (`rag_retriever.py`)

Dual-index, dual-stage retrieval system that:

1. **Query Understanding**: Maps free-form questions to CUAD categories using zero-shot classification
2. **Metadata Filtering**: Applies filename and/or category filters to narrow search space
3. **Dense Retrieval**: Uses Jina Embeddings v2 with FAISS for semantic similarity matching (same model as stored embeddings)
4. **Sparse Retrieval**: Uses BM25 for keyword-based matching (handles legal terminology well)
5. **Candidate Merging**: Combines and deduplicates results from both retrieval methods
6. **Reranking**: Uses cross-encoder (BGE reranker) to score and reorder candidates
7. **Result Formatting**: Returns structured results with metadata (filename, category, answer, etc.)

### Query Understanding (`query_understanding.py`)

Maps user questions to CUAD categories using:
- **Keyword Matching**: Fast path for common legal terms
- **Zero-Shot Classification**: Semantic similarity using sentence transformers
- **Confidence Scoring**: Returns confidence scores to enable fallback to open retrieval

Supports all 41 CUAD categories including:
- Termination For Convenience
- Non-Compete
- Governing Law
- Anti-Assignment
- Change Of Control
- And 36 more...

### BM25 Index (`bm25_index.py`)

Sparse retrieval with:
- **Metadata Filtering**: Filter by filename or category before searching
- **Caching**: Pickle-based caching for fast index loading
- **Global Cache**: Maintains in-memory cache to avoid repeated loading

### Reranker (`reranker.py`)

Cross-encoder reranking using:
- **BGE Reranker**: `BAAI/bge-reranker-base` model
- **CPU Mode**: Configured to run on CPU to avoid CUDA compatibility issues
- **Lazy Loading**: Models loaded only when needed

### Question Answering (`qa_assistant.py`)

Uses retrieved clauses as context for GPT-4o-mini to generate answers with citations.

## Retrieval Stack Architecture

The retrieval system implements a dual-index, dual-stage approach:

### Stage 1: Dual-Index Retrieval

1. **Dense Index (FAISS)**
   - Uses Jina Embeddings v2 (`jinaai/jina-embeddings-v2-base-en`)
   - Indexed with FAISS L2 distance
   - Captures semantic similarity
   - Good for conceptual queries
   - Same model used for both indexing and query encoding for optimal matching

2. **Sparse Index (BM25)**
   - Keyword-based retrieval
   - Excellent for legal terminology ("governed by", "termination for convenience", etc.)
   - Handles exact phrase matching
   - Fast with cached index

### Stage 2: Reranking

- Cross-encoder reranker (BGE) scores all candidate pairs
- Provides fine-grained relevance scoring
- Returns top-k most relevant results

### Metadata Filters

- **Filename Filter**: When user uploads a single contract, filter to that contract only
- **Category Filter**: When query maps to a CUAD category, optionally filter to that category
- Filters applied before retrieval to improve precision

### Query Understanding

- Automatic category detection from free-form questions
- Example: "Can they end this without cause?" → "Termination For Convenience"
- Falls back to open retrieval if confidence is low

## Configuration

Key constants in `consts.py`:
- `CUAD_CSV_FILEPATH`: Path to CUAD master CSV
- `CUAD_JSON_FILEPATH`: Path to CUAD JSON file
- `FULL_CONTRACTS_TXT_DIR`: Directory containing full contract text files

Default paths:
- Embeddings: `processed_data/jina_embeddings.npy`
- BM25 cache: `processed_data/bm25_index.pkl` and `bm25_df.pkl`
- Prepared data: `cuad_prepared_data/cuad_long_clauses.parquet`

## CUAD Categories

The system supports all 41 CUAD categories for contract clause extraction:

**Dates & Terms**: Agreement Date, Effective Date, Expiration Date, Renewal Term, Notice Period To Terminate Renewal

**Parties & Identification**: Document Name, Parties

**Legal Framework**: Governing Law, Most Favored Nation

**Restrictions**: Non-Compete, Exclusivity, No-Solicit Of Customers, No-Solicit Of Employees, Competitive Restriction Exception, Non-Disparagement

**Termination & Assignment**: Termination For Convenience, Change Of Control, Anti-Assignment

**Intellectual Property**: Ip Ownership Assignment, Joint Ip Ownership, License Grant, Non-Transferable License, Affiliate License-Licensor, Affiliate License-Licensee, Unlimited/All-You-Can-Eat-License, Irrevocable Or Perpetual License

**Financial Terms**: Revenue/Profit Sharing, Price Restrictions, Minimum Commitment, Volume Restriction

**Liability & Risk**: Uncapped Liability, Cap On Liability, Liquidated Damages, Warranty Duration, Insurance, Covenant Not To Sue

**Other**: Rofr/Rofo/Rofn, Source Code Escrow, Post-Termination Services, Audit Rights, Third Party Beneficiary

## Performance Notes

- **First Query**: May take longer due to model loading (embeddings, reranker)
- **Subsequent Queries**: Fast due to lazy loading and caching
- **BM25 Index**: Built once and cached; subsequent loads are instant
- **CPU Mode**: All models configured for CPU to avoid CUDA compatibility issues


