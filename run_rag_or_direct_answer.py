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


def build_rag_prompt(question: str, context_chunks: List[Dict]) -> str:
    """
    Build a RAG prompt given retrieved chunks.

    context_chunks: list of dicts with keys 'text', 'score', 'filename', 'chunk_id'
    """
    context_str_parts = []
    for i, c in enumerate(context_chunks, start=1):
        context_str_parts.append(
            f"[Chunk {i} | file={c['filename']} | chunk_id={c['chunk_id']} | score={c['score']:.3f}]\n"
            f"{c['text']}\n"
        )
    context_str = "\n\n".join(context_str_parts)

    prompt = f"""
You are a precise legal contracts assistant.

You will be given several excerpts from a contract and a user question.
Use ONLY the information in the excerpts to answer the question.
If the answer cannot be found in the excerpts, say that the information is not clearly specified in the provided text.

Excerpts:
{context_str}

Question: {question}

Answer (be concise but specific, and cite which chunks you used in natural language):
"""
    return prompt.strip()


def build_direct_prompt(question: str) -> str:
    prompt = f"""
You are a legal contracts assistant.

Answer the following question about typical commercial contracts as accurately as you can.
If the question is ambiguous, explain the uncertainty.

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


def retrieve_chunks(
    index,
    embedder: SentenceTransformer,
    question: str,
    contract_filename: str,
    top_k: int = TOP_K,
) -> List[Dict]:
    """
    Retrieve top_k chunks from Pinecone for a given question, restricted to one contract filename.
    """
    q_emb = embedder.encode(question, convert_to_numpy=True).tolist()

    res = index.query(
        vector=q_emb,
        top_k=top_k,
        include_metadata=True,
        filter={"filename": {"$eq": contract_filename}},
    )

    chunks = []
    for m in res.get("matches", []):
        meta = m.get("metadata", {})
        chunks.append(
            {
                "text": meta.get("text", ""),
                "filename": meta.get("filename", contract_filename),
                "chunk_id": meta.get("chunk_id", -1),
                "score": m.get("score", 0.0),
            }
        )
    return chunks


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
        company = str(row["Company"])          # TODO: Rename to filename
        question = str(row["Question"])         # TODO: Rename to question_template if this is the correct column name
        right_answer = str(row["Right Answer"]) # TODO: Rename to answer if this is the correct column name
        context = str(row.get("Context", ""))   # TODO: Rename to context if this is the correct column name

        # --- RAG mode ---
        chunks = retrieve_chunks(index, embedder, question, contract_filename=company, top_k=TOP_K)
        similarity_scores = [c["score"] for c in chunks]
        rag_prompt = build_rag_prompt(question, chunks) if chunks else build_rag_prompt(question, [])
        rag_answer = call_gemini(gemini, rag_prompt)
        time.sleep(SLEEP_BETWEEN_CALLS)

        # --- Direct mode ---
        direct_prompt = build_direct_prompt(question)
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
                "Similarity Score": str(similarity_scores),
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
