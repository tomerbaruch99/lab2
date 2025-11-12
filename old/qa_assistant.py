import os
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from vectorstore import load_vectorstore

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBEDDER = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
VS = load_vectorstore()

def answer_question(query, k=5):
    q_emb = EMBEDDER.encode([query])
    retrieved, _ = VS.search(q_emb, k=k)

    context = "\n\n".join(
        f"[{r.filename} - {r.clause_type}]\n{r.clause_text}"
        for _, r in retrieved.iterrows()
    )

    prompt = f"""
You are a contract lawyer AI. Answer the user query using only the text below.

Relevant Clauses:
{context}

Question:
{query}

Answer clearly and cite which clause(s) you used.
"""

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)

    if hasattr(response, "text") and response.text:
        return response.text

    # Fallback: extract first non-empty candidate content.
    for candidate in getattr(response, "candidates", []):
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", []) or []
        texts = [
            getattr(part, "text", None)
            for part in parts
            if getattr(part, "text", None)
        ]
        if texts:
            return "\n".join(texts)

    raise RuntimeError("Gemini returned an empty response.")
