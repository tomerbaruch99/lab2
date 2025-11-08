# Contract Clause Q&A Assistant

A Retrieval-Augmented Generation (RAG) system for answering questions about contract clauses using the CUAD (Contract Understanding Atticus Dataset) dataset. The system combines dense and sparse retrieval methods with reranking to provide accurate answers to legal questions.

## Features

- **Data Preparation Pipeline**: Processes CUAD dataset into structured format with train/val/test splits
- **Dual Retrieval System**: 
  - Dense retrieval using Legal-BERT embeddings with FAISS
  - Sparse retrieval using BM25
- **Reranking**: Uses BGE reranker to improve retrieval quality
- **Question Answering**: GPT-4o-mini powered Q&A with citation support
- **Embedding Generation**: Support for multiple embedding models (Jina, Legal-BERT, MPNet)

## Project Structure

```
.
├── app.py                      # Main CLI application
├── qa_assistant.py            # Question answering interface
├── rag_retriever.py           # Hybrid retrieval system (dense + sparse + rerank)
├── embeddings.py              # Embedding generation utilities
├── vectorstore.py             # FAISS-based vector store
├── bm25_index.py              # BM25 sparse retrieval
├── reranker.py                # BGE reranker integration
├── data_preparation.py        # CUAD data processing pipeline
├── consts.py                  # Configuration constants
├── data/                      # CUAD dataset files
│   └── CUAD_v1/
├── cuad_prepared_data/        # Processed data outputs
└── processed_data/            # Additional processed outputs
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

## Usage

### Interactive Q&A

Run the main application:
```bash
python app.py
```

Example questions:
- "Does this agreement restrict assignment?"
- "What are the non-compete restrictions?"
- "What is the termination clause?"

### Programmatic Usage

```python
from qa_assistant import answer_question

answer = answer_question("Does this agreement restrict assignment?")
print(answer)
```

### Retrieval Only

```python
from rag_retriever import retrieve

results = retrieve("non-compete clause", k_dense=8, k_bm25=8, final_k=5)
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

Hybrid retrieval system that:
1. Performs dense retrieval using Legal-BERT + FAISS
2. Performs sparse retrieval using BM25
3. Merges and deduplicates results
4. Reranks candidates using BGE reranker
5. Returns top-k results

### Question Answering (`qa_assistant.py`)

Uses retrieved clauses as context for GPT-4o-mini to generate answers with citations.

## Configuration

Key constants in `consts.py`:
- `CUAD_CSV_FILEPATH`: Path to CUAD master CSV
- `CUAD_JSON_FILEPATH`: Path to CUAD JSON file
- `FULL_CONTRACTS_TXT_DIR`: Directory containing full contract text files

## License

This project uses the CUAD dataset. Please refer to the [Atticus Project](https://www.atticusproject.org/) for dataset licensing information.

