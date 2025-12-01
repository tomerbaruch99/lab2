"""
Runs two modes for each test question:
- Direct answer (no retrieval)
- RAG answer (Pinecone retrieval + Gemini)

Input:
    eval_data/testset.xlsx  (built by infer_testset_from_parquet.py)

Output:
    model_responses/gemini-2.5-flash-testset.xlsx
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import pandas as pd
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# ----------------- Paths & constants -----------------

TESTSET_PATH = Path("eval_data/testset.parquet")
MODEL_RESPONSES_DIR = Path("model_responses")
MODEL_RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

API_KEYS_PATH = Path("api_keys.json")
PINECONE_INDEX_NAME = "contracts-recursive-index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

TOP_K = 8          # number of chunks to retrieve per question
SLEEP_BETWEEN_CALLS = 3.0  # seconds; adjust to rate limits
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 10.0  # seconds


# ----------------- Helpers -----------------

def load_api_keys(api_keys_path: Path) -> Dict[str, str]:
    with api_keys_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def init_gemini(api_keys: Dict[str, str]):
    gemini_api_key = api_keys["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
    return model


def init_pinecone(api_keys: Dict[str, str]) -> Any:
    pinecone_api_key = api_keys["PINECONE_API_KEY"]
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(PINECONE_INDEX_NAME)
    return index


def init_embedder() -> SentenceTransformer:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return model


def build_rag_prompt(
    question: str,
    training_chunks: List[Dict],
    test_document_chunks: List[Dict] = None,
) -> str:
    """
    Build a RAG prompt with chunks from training documents (knowledge base) and test document.
    
    Args:
        question: User's question
        training_chunks: Chunks retrieved from training documents (examples/patterns)
        test_document_chunks: Chunks from the test document being asked about
    """
    # Build training examples section
    training_str_parts = []
    if training_chunks:
        training_str_parts.append("=== Examples from Training Documents (for reference) ===")
        for i, c in enumerate(training_chunks, start=1):
            training_str_parts.append(
                f"[Training Example {i} | file={c.get('filename', 'unknown')} | score={c.get('score', 0.0):.3f}]\n"
                f"{c['text']}\n"
            )
    
    # Build test document section
    test_doc_str_parts = []
    if test_document_chunks:
        test_doc_str_parts.append("=== Excerpts from the Document You Are Analyzing ===")
        for i, c in enumerate(test_document_chunks, start=1):
            test_doc_str_parts.append(
                f"[Document Chunk {i} | chunk_id={c.get('chunk_id', i-1)}]\n"
                f"{c['text']}\n"
            )
    
    # Combine sections
    all_context = []
    if training_str_parts:
        all_context.append("\n\n".join(training_str_parts))
    if test_doc_str_parts:
        all_context.append("\n\n".join(test_doc_str_parts))
    
    context_str = "\n\n".join(all_context) if all_context else "No context provided."

    prompt = f"""
You are a precise legal contracts assistant trained on legal contract examples.

You will be given:
1. Examples from training documents (for reference on how to extract information)
2. Excerpts from the document you need to analyze
3. A question about the document

Use the training examples to understand what kind of information to look for, then answer the question using ONLY the information from the document excerpts.
If the answer cannot be found in the document excerpts, say that the information is not clearly specified in the provided text.

{context_str}

Question: {question}

Answer (be concise but specific, and cite which document chunks you used):
"""
    return prompt.strip()


def build_direct_prompt(question: str, test_document_text: str = None) -> str:
    """
    Build a prompt for direct answer mode (no retrieval from training).
    This is a baseline to compare against RAG mode.
    
    Args:
        question: The question to answer
        test_document_text: Full text of the test document
    """
    if test_document_text:
        # Truncate if too long (keep first 8000 chars to fit in prompt)
        doc_preview = test_document_text[:8000]
        if len(test_document_text) > 8000:
            doc_preview += "\n\n[Document truncated - showing first 8000 characters]"
        
        prompt = f"""
You are a legal contracts assistant.

You will be given a legal document and a question about it.
Answer the question using ONLY the information in the document.
If the answer cannot be found in the document, say that the information is not clearly specified in the provided text.

Document:
{doc_preview}

Question: {question}

Answer:
"""
    else:
        prompt = f"""
You are a legal contracts assistant.

Answer the following question about a legal contract as accurately as you can based on your general knowledge.
If the question is about a specific document and you don't have access to it, state that you cannot answer without seeing the document.

Question: {question}

Answer:
"""
    return prompt.strip()


def call_gemini(model, prompt: str, retry_count: int = 0) -> str:
    """
    Call Gemini API with retry logic for rate limiting.
    """
    try:
        response = model.generate_content(prompt)
        # response.text is usually fine; add a defensive fallback
        return getattr(response, "text", "").strip()
    except (google_exceptions.ResourceExhausted, google_exceptions.RetryError) as e:
        # Handle rate limiting and quota exhaustion
        if retry_count < MAX_RETRIES:
            # Exponential backoff: 10s, 20s, 40s, 80s, 160s
            delay = INITIAL_RETRY_DELAY * (2 ** retry_count)
            print(f"[WARN] Rate limit hit. Waiting {delay:.1f}s before retry {retry_count + 1}/{MAX_RETRIES}...")
            time.sleep(delay)
            return call_gemini(model, prompt, retry_count + 1)
        else:
            raise Exception(f"Rate limit exceeded after {MAX_RETRIES} retries. Please wait and try again later.") from e
    except Exception as e:
        error_str = str(e).lower()
        # Fallback check for rate limit errors in string representation
        if "quota" in error_str or "rate_limit" in error_str or "429" in error_str or "resourceexhausted" in error_str:
            if retry_count < MAX_RETRIES:
                delay = INITIAL_RETRY_DELAY * (2 ** retry_count)
                print(f"[WARN] Rate limit hit. Waiting {delay:.1f}s before retry {retry_count + 1}/{MAX_RETRIES}...")
                time.sleep(delay)
                return call_gemini(model, prompt, retry_count + 1)
            else:
                raise Exception(f"Rate limit exceeded after {MAX_RETRIES} retries. Please wait and try again later.") from e
        else:
            # Re-raise non-rate-limit errors
            raise


def retrieve_chunks_from_training(
    index,
    embedder: SentenceTransformer,
    question: str,
    top_k: int = TOP_K,
) -> List[Dict]:
    """
    Retrieve top_k chunks from TRAINING documents in Pinecone based on semantic similarity.
    This is the knowledge base that helps the model understand how to extract information.
    
    Note: No filename filter - retrieves from all training documents based on question similarity.
    """
    q_emb = embedder.encode(question, convert_to_numpy=True).tolist()
    
    # Retrieve from training documents (no filename filter)
    res = index.query(
        vector=q_emb,
        top_k=top_k,
        include_metadata=True,
    )
    
    chunks = []
    for m in res.get("matches", []):
        meta = m.get("metadata", {})
        chunks.append(
            {
                "text": meta.get("text", ""),
                "filename": meta.get("filename", "unknown"),
                "chunk_id": meta.get("chunk_id", -1),
                "score": m.get("score", 0.0),
            }
        )
    
    return chunks


def load_and_chunk_test_document(document_path: str) -> List[Dict]:
    """
    Load a test document and chunk it for inclusion in the prompt.
    
    Handles filename mismatches:
    - CSV has .pdf filenames, but TXT files have .txt extension
    - Files are in data/CUAD_v1/full_contract_txt/ directory
    
    Returns:
        List of chunk dicts with keys: text, chunk_id, start_char
    """
    from data_preparation import chunk_text, clean_clause_text
    from consts import FULL_CONTRACTS_TXT_DIR
    import os
    
    # Try multiple filename variations
    candidates = []
    original_path = Path(document_path)
    
    # 1. Try as-is
    candidates.append(original_path)
    
    # 2. Try in data directory as-is
    if FULL_CONTRACTS_TXT_DIR:
        candidates.append(Path(FULL_CONTRACTS_TXT_DIR) / original_path.name)
    
    # 3. Try replacing .pdf/.PDF with .txt (case-insensitive)
    if original_path.suffix.lower() == ".pdf":
        stem = original_path.stem
        candidates.append(original_path.with_suffix(".txt"))
        if FULL_CONTRACTS_TXT_DIR:
            candidates.append(Path(FULL_CONTRACTS_TXT_DIR) / f"{stem}.txt")
            # Also try with normalized special characters
            stem_normalized = stem.replace("&", "_")
            candidates.append(Path(FULL_CONTRACTS_TXT_DIR) / f"{stem_normalized}.txt")
    
    # 4. Try adding .txt extension
    candidates.append(Path(str(original_path) + ".txt"))
    if FULL_CONTRACTS_TXT_DIR:
        candidates.append(Path(FULL_CONTRACTS_TXT_DIR) / f"{original_path.name}.txt")
        # Also try with normalized name
        name_normalized = original_path.name.replace("&", "_")
        candidates.append(Path(FULL_CONTRACTS_TXT_DIR) / name_normalized.replace(".PDF", ".txt").replace(".pdf", ".txt"))
    
    # 5. Try just the stem + .txt in data directory (with and without normalization)
    if FULL_CONTRACTS_TXT_DIR:
        candidates.append(Path(FULL_CONTRACTS_TXT_DIR) / f"{original_path.stem}.txt")
        stem_normalized = original_path.stem.replace("&", "_")
        candidates.append(Path(FULL_CONTRACTS_TXT_DIR) / f"{stem_normalized}.txt")
    
    # Try to find existing file
    doc_path = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            doc_path = candidate
            break
    
    # If still not found, try fuzzy matching in data directory
    if not doc_path and FULL_CONTRACTS_TXT_DIR:
        data_dir = Path(FULL_CONTRACTS_TXT_DIR)
        if data_dir.exists():
            # Normalize the stem for fuzzy matching
            # Replace common special characters that might differ
            stem_normalized = original_path.stem.lower()
            # Common substitutions: & -> _, spaces, etc.
            stem_normalized = stem_normalized.replace("&", "_").replace(" ", "_")
            
            # Search for files with matching stem (case-insensitive, fuzzy)
            best_match = None
            best_score = 0
            
            # Get all txt files in directory
            txt_files = list(data_dir.glob("*.txt"))
            if not txt_files:
                # Try case-insensitive glob
                txt_files = [f for f in data_dir.iterdir() if f.suffix.lower() == ".txt" and f.is_file()]
            
            for file_path in txt_files:
                file_stem = file_path.stem.lower()
                file_stem_normalized = file_stem.replace("&", "_").replace(" ", "_")
                
                # Exact match after normalization
                if file_stem_normalized == stem_normalized:
                    doc_path = file_path
                    break
                
                # Fuzzy match: check if stems are similar (one contains the other or vice versa)
                # This handles cases where filenames have slight variations
                if stem_normalized in file_stem_normalized or file_stem_normalized in stem_normalized:
                    # Calculate similarity score (length of common substring)
                    common_len = min(len(stem_normalized), len(file_stem_normalized))
                    if common_len > best_score:
                        best_score = common_len
                        best_match = file_path
            
            # Use best match if no exact match found and similarity is high enough
            if not doc_path and best_match and best_score > len(stem_normalized) * 0.8:
                doc_path = best_match
    
    if not doc_path:
        raise FileNotFoundError(
            f"Document not found: {document_path}\n"
            f"Tried: {[str(c) for c in candidates[:5]]}..."
        )
    
    # Read document
    try:
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            text = doc_path.read_text(encoding="latin-1", errors="ignore")
        except Exception as e:
            raise Exception(f"Could not read document: {e}")
    
    if not text.strip():
        return []
    
    # Clean and chunk
    text = clean_clause_text(text)
    chunks = chunk_text(text, max_chars=4000, overlap=400)
    
    # Convert to list of dicts
    chunk_dicts = []
    for chunk_id, (start_char, chunk_text) in enumerate(chunks):
        chunk_dicts.append({
            "text": chunk_text,
            "chunk_id": chunk_id,
            "start_char": start_char,
        })
    
    return chunk_dicts


def main():
    # 1) Load testset
    if not TESTSET_PATH.exists():
        raise FileNotFoundError(f"Testset not found at {TESTSET_PATH}. Run infer_testset_from_parquet.py first.")
    testset = pd.read_parquet(TESTSET_PATH)

    # 2) Init APIs and models
    api_keys = load_api_keys(API_KEYS_PATH)
    gemini = init_gemini(api_keys)
    index = init_pinecone(api_keys)
    embedder = init_embedder()

    results = []
    for _, row in tqdm(testset.iterrows(), total=len(testset), desc="Evaluating questions"):
        company = str(row["Company"])          # This is the test document filename
        question = str(row["Question"])
        right_answer = str(row["Right Answer"])
        context = str(row.get("Context", ""))

        # --- Load test document ---
        try:
            test_document_chunks = load_and_chunk_test_document(company)
            # Get full document text for direct mode
            test_document_text = "\n\n".join([c["text"] for c in test_document_chunks])
        except Exception as e:
            print(f"[WARN] Could not load test document {company}: {e}")
            test_document_chunks = []
            test_document_text = None

        # --- RAG mode ---
        # Retrieve relevant chunks from TRAINING documents (knowledge base)
        training_chunks = retrieve_chunks_from_training(index, embedder, question, top_k=TOP_K)
        training_similarity_scores = [c["score"] for c in training_chunks]
        
        if not training_chunks:
            print(f"[WARN] No chunks retrieved from training documents for question: {question[:50]}...")
        
        # Build RAG prompt with training chunks (examples) + test document chunks
        rag_prompt = build_rag_prompt(
            question,
            training_chunks=training_chunks,
            test_document_chunks=test_document_chunks[:TOP_K] if test_document_chunks else None
        )
        rag_answer = call_gemini(gemini, rag_prompt)
        time.sleep(SLEEP_BETWEEN_CALLS)

        # --- Direct mode ---
        # Direct mode: answer using test document only (no training retrieval)
        direct_prompt = build_direct_prompt(question, test_document_text=test_document_text)
        direct_answer = call_gemini(gemini, direct_prompt)
        time.sleep(SLEEP_BETWEEN_CALLS)

        results.append(
            {
                "Company": company,
                "Question": question,
                "Right Answer": right_answer,
                "Context": context,
                "RAG Answer": rag_answer,
                "Direct Answer": direct_answer,
                # we store similarity scores as a stringified list to fit your existing code
                "Similarity Score": str(training_similarity_scores),
                # you only have one index right now; keep this string to stay compatible
                "Optimal Index": PINECONE_INDEX_NAME,
                "rag_model": GEMINI_MODEL_NAME,
                "index": PINECONE_INDEX_NAME,
                "additional_flags": "none",
                "optimizer": False,
            }
        )

    out_path = MODEL_RESPONSES_DIR / f"{GEMINI_MODEL_NAME}-testset.parquet"
    out_df = pd.DataFrame(results)
    out_df.to_parquet(out_path, index=False)
    print(f"[OK] Saved model responses to: {out_path}")


if __name__ == "__main__":
    main()
