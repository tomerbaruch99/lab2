# Workflow Analysis and Bug Fix

## Correct Workflow for Legal Document Q&A

**IMPORTANT**: The system should work as follows:

1. **Training/Indexing Phase**: Index training contracts (optional - for reference/knowledge base)
2. **Inference Phase**: 
   - User provides a **NEW** legal document (not in training set)
   - User asks a question about that document
   - System chunks the new document on-the-fly
   - System retrieves relevant chunks from the new document
   - System answers the question using those chunks

**The current `run_rag_or_direct_answer.py` is for evaluation only** - it tests on pre-indexed contracts. For actual inference on new documents, use `infer_new_document.py`.

## Project Workflow Overview

This project implements a RAG (Retrieval-Augmented Generation) system for answering questions about legal contracts from the CUAD dataset. The workflow consists of several stages:

### 1. Data Preparation (`data_preparation.py`)
- **Input**: CUAD master CSV file with 83 columns (one per category)
- **Process**:
  - Flattens wide CSV into long table: (filename, category, context, answer, question_template)
  - Chunks full-contract TXT files into overlapping paragraph windows
  - Creates train/val/test splits at contract level
- **Output**:
  - `cuad_long_clauses.parquet`: Flattened clause data for RAG
  - `cuad_paragraph_index.parquet`: Chunked contract paragraphs for indexing
  - Both files use `filename` from CSV "Filename" column

### 2. Indexing (`indexing_data.py`)
- **Input**: `cuad_paragraph_index.parquet` (from step 1)
- **Process**:
  - Reads paragraph chunks with metadata: (doc_id, filename, chunk_id, start_char, text)
  - Embeds chunks using SentenceTransformer (`all-MiniLM-L6-v2`)
  - Stores vectors in Pinecone with metadata
- **Output**: Pinecone index `contracts-recursive-index`
- **Note**: Filenames stored are from TXT files (`p.name` = full filename with .txt extension)

### 3. Testset Creation (`infer_testset_from_parquet.py`)
- **Input**: `cuad_long_clauses.parquet` (from step 1)
- **Process**:
  - Samples 200 QA pairs from test split
  - Maps to testset format: (Filename, Question, Right Answer, Context)
- **Output**: `eval_data/testset.parquet`
- **Note**: "Filename" column contains CSV filenames (e.g., "file.pdf")

### 4a. RAG Inference for New Documents (`infer_new_document.py`) ⭐ **USE THIS FOR ACTUAL INFERENCE**
- **Input**: 
  - New legal document (TXT file) - **not in training set**
  - Question about the document
- **Process**:
  1. Loads and chunks the new document on-the-fly
  2. Embeds question and document chunks
  3. Retrieves top-k chunks from the document using semantic similarity
  4. Builds RAG prompt with retrieved chunks
  5. Calls Gemini to generate answer
- **Output**: Answer to the question
- **Usage**: 
  ```bash
  python infer_new_document.py --document_path <document.txt> --question "What is the name of this agreement?"
  ```

### 4b. RAG Evaluation (`run_rag_or_direct_answer.py`) - **FOR EVALUATION ONLY**
- **Input**: `eval_data/testset.parquet` (from step 3)
- **Process**:
  - For each question:
    1. Retrieves top-k chunks from Pinecone filtered by filename
    2. Builds RAG prompt with retrieved chunks
    3. Calls Gemini to generate answer
    4. Also generates direct answer (no retrieval) for comparison
- **Output**: `model_responses/gemini-2.5-flash-testset.parquet`
- **Note**: This is for evaluating the system on test set, not for actual inference on new documents

### 5. Evaluation (`evaluate_rag_or_direct.py`)
- **Input**: `model_responses/*.parquet` (from step 4)
- **Process**:
  - Uses Gemini to evaluate each answer on 5 criteria:
    - Factuality, Relevance, Completeness, Confidence, Correctness
- **Output**: `evaluation/*_evaluation.csv`


## Code Files Summary

### Main Scripts
- `data_preparation.py`: ETL pipeline for CUAD data
- `indexing_data.py`: Indexes chunks into Pinecone
- `infer_testset_from_parquet.py`: Creates evaluation testset
- `run_rag_or_direct_answer.py`: Runs RAG inference
- `evaluate_rag_or_direct.py`: Evaluates model responses

### Configuration
- `consts.py`: Constants and configuration
- `api_keys.json`: API keys (not in repo)

### Data Files
- `data/CUAD_v1/`: Raw CUAD dataset
- `cuad_prepared_data/`: Processed data (long clauses, paragraph index)
- `eval_data/`: Testset for evaluation
- `model_responses/`: Model outputs
- `evaluation/`: Evaluation results

## Input/Output Flow

```
CSV (Filename column)
  ↓
data_preparation.py
  ↓
cuad_long_clauses.parquet (filename from CSV)
  ↓
infer_testset_from_parquet.py
  ↓
testset.parquet (Filename = CSV filename)
  ↓
run_rag_or_direct_answer.py
  ↓
Pinecone (filename from TXT)
  ↓
Chunks retrieved ✓
  ↓
Gemini generates answer
```
