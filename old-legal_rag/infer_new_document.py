"""
Inference script for new legal documents.

This script handles the proper workflow:
1. User provides a new legal document (not in training set)
2. User asks a question about that document
3. System chunks the document and retrieves relevant chunks from it
4. System uses retrieved chunks to answer the question

Usage:
    python infer_new_document.py --document_path <path_to_document.txt> --question "What is the name of this agreement?"
    
Or for batch processing:
    python infer_new_document.py --input_csv <questions.csv> --output_csv <answers.csv>
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Import chunking function from data_preparation
from data_preparation import chunk_text, clean_clause_text

# ----------------- Paths & constants -----------------

API_KEYS_PATH = Path("api_keys.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

TOP_K = 8          # number of chunks to retrieve per question
SLEEP_BETWEEN_CALLS = 3.0  # seconds; adjust to rate limits
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 10.0  # seconds
CHUNK_CHARS = 4000  # characters per chunk
CHUNK_OVERLAP = 400  # overlap between chunks


# ----------------- Helpers -----------------

def load_api_keys(api_keys_path: Path) -> Dict[str, str]:
    with api_keys_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def init_gemini(api_keys: Dict[str, str]):
    gemini_api_key = api_keys["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
    return model


def init_embedder() -> SentenceTransformer:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return model


def load_and_chunk_document(document_path: str) -> List[Dict[str, Any]]:
    """
    Load a legal document and chunk it into overlapping windows.
    
    Returns:
        List of chunk dicts with keys: text, chunk_id, start_char
    """
    doc_path = Path(document_path)
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")
    
    # Read document
    try:
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            text = doc_path.read_text(encoding="latin-1", errors="ignore")
        except Exception as e:
            raise Exception(f"Could not read document: {e}")
    
    if not text.strip():
        raise ValueError(f"Document is empty: {document_path}")
    
    # Clean and chunk
    text = clean_clause_text(text)
    chunks = chunk_text(text, max_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP)
    
    # Convert to list of dicts
    chunk_dicts = []
    for chunk_id, (start_char, chunk_text) in enumerate(chunks):
        chunk_dicts.append({
            "text": chunk_text,
            "chunk_id": chunk_id,
            "start_char": start_char,
        })
    
    return chunk_dicts


def retrieve_chunks_from_document(
    embedder: SentenceTransformer,
    question: str,
    document_chunks: List[Dict[str, Any]],
    top_k: int = TOP_K,
) -> List[Dict]:
    """
    Retrieve top_k chunks from a document based on semantic similarity to the question.
    
    Args:
        embedder: SentenceTransformer model
        question: User's question
        document_chunks: List of chunk dicts from the document
        top_k: Number of chunks to retrieve
    
    Returns:
        List of chunk dicts with added 'score' field, sorted by relevance
    """
    if not document_chunks:
        return []
    
    # Embed question
    q_emb = embedder.encode(question, convert_to_numpy=True)
    
    # Embed all chunks
    chunk_texts = [chunk["text"] for chunk in document_chunks]
    chunk_embs = embedder.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)
    
    # Compute cosine similarity
    import numpy as np
    # Normalize embeddings
    q_emb_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
    chunk_embs_norm = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarities
    similarities = np.dot(chunk_embs_norm, q_emb_norm)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Build result list
    retrieved_chunks = []
    for idx in top_indices:
        chunk = document_chunks[idx].copy()
        chunk["score"] = float(similarities[idx])
        retrieved_chunks.append(chunk)
    
    # Sort by score (already sorted, but ensure)
    retrieved_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    return retrieved_chunks


def build_rag_prompt(question: str, context_chunks: List[Dict]) -> str:
    """
    Build a RAG prompt given retrieved chunks from the document.
    
    context_chunks: list of dicts with keys 'text', 'score', 'chunk_id'
    """
    context_str_parts = []
    for i, c in enumerate(context_chunks, start=1):
        context_str_parts.append(
            f"[Chunk {i} | chunk_id={c['chunk_id']} | score={c['score']:.3f}]\n"
            f"{c['text']}\n"
        )
    context_str = "\n\n".join(context_str_parts)

    prompt = f"""
You are a precise legal contracts assistant.

You will be given several excerpts from a legal contract and a user question.
Use ONLY the information in the excerpts to answer the question.
If the answer cannot be found in the excerpts, say that the information is not clearly specified in the provided text.

Excerpts from the contract:
{context_str}

Question: {question}

Answer (be concise but specific, and cite which chunks you used in natural language):
"""
    return prompt.strip()


def call_gemini(model, prompt: str, retry_count: int = 0) -> str:
    """
    Call Gemini API with retry logic for rate limiting.
    """
    try:
        response = model.generate_content(prompt)
        return getattr(response, "text", "").strip()
    except (google_exceptions.ResourceExhausted, google_exceptions.RetryError) as e:
        if retry_count < MAX_RETRIES:
            delay = INITIAL_RETRY_DELAY * (2 ** retry_count)
            print(f"[WARN] Rate limit hit. Waiting {delay:.1f}s before retry {retry_count + 1}/{MAX_RETRIES}...")
            time.sleep(delay)
            return call_gemini(model, prompt, retry_count + 1)
        else:
            raise Exception(f"Rate limit exceeded after {MAX_RETRIES} retries.") from e
    except Exception as e:
        error_str = str(e).lower()
        if "quota" in error_str or "rate_limit" in error_str or "429" in error_str or "resourceexhausted" in error_str:
            if retry_count < MAX_RETRIES:
                delay = INITIAL_RETRY_DELAY * (2 ** retry_count)
                print(f"[WARN] Rate limit hit. Waiting {delay:.1f}s before retry {retry_count + 1}/{MAX_RETRIES}...")
                time.sleep(delay)
                return call_gemini(model, prompt, retry_count + 1)
            else:
                raise Exception(f"Rate limit exceeded after {MAX_RETRIES} retries.") from e
        else:
            raise


def answer_question_for_document(
    document_path: str,
    question: str,
    gemini_model,
    embedder: SentenceTransformer,
) -> Dict[str, Any]:
    """
    Process a single question about a document.
    
    Returns:
        Dict with keys: question, answer, chunks_retrieved, similarity_scores
    """
    # Load and chunk document
    document_chunks = load_and_chunk_document(document_path)
    
    if not document_chunks:
        return {
            "question": question,
            "answer": "Error: Could not process document (empty or unreadable).",
            "chunks_retrieved": 0,
            "similarity_scores": [],
        }
    
    # Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks_from_document(
        embedder, question, document_chunks, top_k=TOP_K
    )
    
    # Build prompt and get answer
    if retrieved_chunks:
        rag_prompt = build_rag_prompt(question, retrieved_chunks)
        answer = call_gemini(gemini_model, rag_prompt)
        time.sleep(SLEEP_BETWEEN_CALLS)
    else:
        answer = "Error: No relevant chunks could be retrieved from the document."
        rag_prompt = None
    
    similarity_scores = [c["score"] for c in retrieved_chunks]
    
    return {
        "question": question,
        "answer": answer,
        "chunks_retrieved": len(retrieved_chunks),
        "similarity_scores": similarity_scores,
        "document_path": document_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Answer questions about a new legal document"
    )
    parser.add_argument(
        "--document_path",
        type=str,
        help="Path to the legal document (TXT file)",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question about the document",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        help="CSV file with columns: document_path, question (for batch processing)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="inference_results.csv",
        help="Output CSV file for results",
    )
    args = parser.parse_args()
    
    # Validate arguments
    if not args.document_path and not args.input_csv:
        parser.error("Either --document_path or --input_csv must be provided")
    if args.document_path and not args.question:
        parser.error("--question is required when using --document_path")
    
    # Initialize models
    print("[STEP] Initializing models...")
    api_keys = load_api_keys(API_KEYS_PATH)
    gemini = init_gemini(api_keys)
    embedder = init_embedder()
    print("[OK] Models initialized")
    
    # Process single question or batch
    if args.document_path:
        # Single question mode
        print(f"\n[STEP] Processing question about document: {args.document_path}")
        result = answer_question_for_document(
            args.document_path, args.question, gemini, embedder
        )
        print(f"\n[RESULT]")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Chunks retrieved: {result['chunks_retrieved']}")
        print(f"Similarity scores: {result['similarity_scores']}")
    else:
        # Batch mode
        print(f"\n[STEP] Processing batch from: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        
        if "document_path" not in df.columns or "question" not in df.columns:
            raise ValueError("CSV must have 'document_path' and 'question' columns")
        
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
            doc_path = row["document_path"]
            question = row["question"]
            
            try:
                result = answer_question_for_document(
                    doc_path, question, gemini, embedder
                )
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Failed for {doc_path}: {e}")
                results.append({
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "chunks_retrieved": 0,
                    "similarity_scores": [],
                    "document_path": doc_path,
                })
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output_csv, index=False)
        print(f"\n[OK] Saved results to: {args.output_csv}")


if __name__ == "__main__":
    main()

