from sentence_transformers import SentenceTransformer
import openai
from vectorstore.py import load_vectorstore

openai.api_key = os.getenv("OPENAI_API_KEY")
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

    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message["content"]
