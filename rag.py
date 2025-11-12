import os, re, json, argparse, math
from typing import List, Dict, Any, Tuple
from config import *
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from consts import load_experiment_config

from config import (
    CATEGORY_ROUTER_MODE, CATEGORY_HARD_BONUS, CATEGORY_SOFT_MAX_BONUS,
    CATEGORY_QUERY_WEIGHT, CATEGORY_MAX_REPS, FILENAME_MATCH_BONUS
)

# keyword router (extend freely)
CATEGORY_KEYWORDS = {
    "Exclusivity": [
        "exclusivity", "exclusive dealing", "sole supplier", "exclusive license"
    ],
    "Non-Compete": [
        "non-compete", "non compete", "compete restriction", "competition restriction"
    ],
    "Governing Law": [
        "governing law", "jurisdiction", "venue", "choice of law"
    ],
    "Effective Date": [
        "effective date", "effective on", "becomes effective", "commencement date"
    ],
    "Expiration Date": [
        "expire", "expiration date", "term ends", "end of term"
    ],
    "License Grant": [
        "license grant", "licensed rights", "scope of license", "license to use"
    ],
    "Cap On Liability": [
        "liability cap", "cap on liability", "limitation of liability", "aggregate liability"
    ],
    "Uncapped Liability": [
        "uncapped liability", "unlimited liability", "no cap on liability"
    ],
    "Audit Rights": [
        "audit", "audit rights", "inspection rights", "books and records"
    ],
    "Termination For Convenience": [
        "terminate for convenience", "without cause termination", "convenience termination"
    ],
    "Insurance": [
        "insurance", "coverage", "insured", "certificate of insurance"
    ],
    "Warranty Duration": [
        "warranty", "warranty duration", "warranty period", "limited warranty"
    ],
    "Change Of Control": [
        "change of control", "merger", "acquisition", "control changes"
    ],
    "Anti-Assignment": [
        "anti-assignment", "assignment", "may not assign", "no assignment"
    ],
    "Audit Rights": [
        "audit rights", "audit", "inspection", "books and records"
    ],
    # add more CUAD categories as needed...
}

def _route_category_hint(question: str):
    """
    Returns (category_name: str, confidence: float in [0,1], matched_keywords: list).
    Confidence is min(1, matches/2) — 1 keyword -> 0.5; 2+ -> 1.0.
    """
    q = question.lower()
    best_cat, best_hits = "", []
    for cat, kws in CATEGORY_KEYWORDS.items():
        hits = [kw for kw in kws if kw in q]
        if len(hits) > len(best_hits):
            best_cat, best_hits = cat, hits
    if not best_cat:
        return "", 0.0, []
    confidence = min(1.0, len(best_hits) / 2.0)
    return best_cat, confidence, best_hits

def _apply_query_hint(question: str, cat_hint: str, mode: str, confidence: float):
    """
    Returns a modified query string with category hint repeated according to mode.
    """
    if mode == "off" or not cat_hint:
        return question
    # repetitions:
    if mode == "hard":
        reps = CATEGORY_MAX_REPS
    else:  # soft
        # 1 .. CATEGORY_MAX_REPS scaled by confidence and CATEGORY_QUERY_WEIGHT
        import math
        scaled = max(1.0, CATEGORY_QUERY_WEIGHT * (1.0 + confidence * (CATEGORY_MAX_REPS - 1)))
        reps = int(min(CATEGORY_MAX_REPS, math.ceil(scaled)))
    hint = (" Category: " + cat_hint) * reps
    return f"{question}{hint}"


def _faiss_load(path_idx: str):
    import faiss, pickle
    index = faiss.read_index(path_idx)
    with open(path_idx + ".meta.pkl", "rb") as f:
        metas = pickle.load(f)
    return index, metas

def _faiss_search(index, query_vec, k=TOP_K):
    import numpy as np
    q = query_vec.astype("float32")
    # normalize for cosine
    q = q / (np.linalg.norm(q) + 1e-12)
    D, I = index.search(q[None,:], k)
    return I[0].tolist(), D[0].tolist()

def _pinecone_search(index, query_vec, k=TOP_K):
    res = index.query(vector=query_vec.tolist(), top_k=k, include_metadata=True)
    hits = []
    for m in res.matches:
        hits.append((m.metadata, float(m.score)))
    return hits

def _init_llm():
    if LLM_PROVIDER == "openai":
        import openai
        openai.api_key = OPENAI_API_KEY
        def call(messages):
            resp = openai.ChatCompletion.create(model=LLM_MODEL, messages=messages, temperature=0.1)
            return resp.choices[0].message["content"]
        return call
    elif LLM_PROVIDER == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(LLM_MODEL)
        def call(messages):
            # flatten to a single prompt
            role_map = {"system":"","user":"","assistant":""}
            text = "\n".join([m["content"] for m in messages])
            return model.generate_content(text).text
        return call
    elif LLM_PROVIDER == "cohere":
        import cohere
        co = cohere.Client(api_key=COHERE_API_KEY)
        def call(messages):
            text = "\n".join([m["content"] for m in messages])
            r = co.generate(model=LLM_MODEL, prompt=text, temperature=0.1)
            return r.generations[0].text
        return call
    else:
        raise ValueError("Unsupported LLM_PROVIDER")

def _init_reranker():
    if not RERANKER:
        return None
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER)

def _pack_context(chunks: List[Dict[str,Any]], max_tokens: int) -> Tuple[str, List[Dict[str,Any]]]:
    """
    Naive token budget by char length. Tag each chunk with [#].
    """
    packed, used = [], []
    budget = max_tokens * 4  # rough char budget
    for i, c in enumerate(chunks, start=1):
        tag = f"[{i}] filename={c['filename']} | category={c['category']}"
        text = f"{tag}\n{c['text']}\n"
        if len("".join(packed)) + len(text) > budget:
            break
        packed.append(text)
        used.append(c)
    return "\n".join(packed), used

def answer(question: str, k: int = TOP_K):
    config = load_experiment_config()
    CATEGORY_ROUTER_MODE = config["category_router_mode"]
    CATEGORY_HARD_BONUS = config["category_hard_bonus"]
    CATEGORY_SOFT_MAX_BONUS = config["category_soft_max_bonus"]
    CATEGORY_QUERY_WEIGHT = config["category_query_weight"]
    CATEGORY_MAX_REPS = config["category_max_reps"]
    FILENAME_MATCH_BONUS = config["filename_match_bonus"]
    RERANKER = config["reranker"]
    EMBED_MODEL = config["embed_model"]
    TOP_K = config["top_k"]
    emb_model = SentenceTransformer(EMBED_MODEL)
    cat_hint, cat_conf, kw_hits = _route_category_hint(question)
    query_for_embedding = _apply_query_hint(question, cat_hint, CATEGORY_ROUTER_MODE, cat_conf)
    qv = emb_model.encode(query_for_embedding)

    # Retrieve
    if config["vstore"] == "pinecone":
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        idx = pc.Index(os.getenv("PINECONE_INDEX"))
        raw_hits = _pinecone_search(idx, qv, k=config["top_k"]*3)  # overfetch for reranker
        candidates = [{"text": h[0]["text"], **{k2:v2 for k2,v2 in h[0].items() if k2!="text"}, "score": h[1]} for h in raw_hits]
    else:
        index, metas = _faiss_load(os.getenv("FAISS_PATH"))
        import numpy as np
        I, D = _faiss_search(index, qv, k=k*3)
        candidates = [{"text": metas[i]["text"], **{k2:v2 for k2,v2 in metas[i].items() if k2!="text"}, "score": float(D[j])} for j,i in enumerate(I)]

    if CATEGORY_ROUTER_MODE in ("hard", "soft") and cat_hint:
        if CATEGORY_ROUTER_MODE == "hard":
            cat_bonus = config["category_hard_bonus"]
        else:
            cat_bonus = config["category_soft_max_bonus"] * cat_conf  # dynamic

        for c in candidates:
            # category bonus
            if str(c.get("category","")).lower() == cat_hint.lower():
                c["score"] = float(c["score"]) + cat_bonus
                c["_cat_bonus_applied"] = cat_bonus
            # tiny filename bonus: if question includes the filename's stem (sometimes user asks per doc)
            fn = str(c.get("filename",""))
            if fn and fn.lower() in question.lower():
                c["score"] = float(c["score"]) + FILENAME_MATCH_BONUS
                c["_fn_bonus_applied"] = FILENAME_MATCH_BONUS

    # Re-sort by shaped score
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Rerank (optional)
    reranker = _init_reranker()
    if reranker:
        pairs = [(question, c["text"]) for c in candidates]
        scores = reranker.predict(pairs).tolist()
        if CATEGORY_ROUTER_MODE in ("hard", "soft") and cat_hint:
            for c in candidates:
                base = c.get("rerank", c["score"])
                add = 0.0
                if str(c.get("category","")).lower() == cat_hint.lower():
                    add += (cat_bonus / 2.0)  # half the pre-rerank push
                if str(c.get("filename","")) and c["filename"].lower() in question.lower():
                    add += (FILENAME_MATCH_BONUS / 2.0)
                c["rerank"] = float(base) + add
            candidates.sort(key=lambda x: x.get("rerank", x["score"]), reverse=True)
        for c, s in zip(candidates, scores):
            c["rerank"] = float(s)
        candidates.sort(key=lambda x: x.get("rerank", x["score"]), reverse=True)
    # Final top_k
    top = candidates[:k]

    packed_context, used = _pack_context(top, MAX_CONTEXT_TOKENS)

    # Call LLM
    call_llm = _init_llm()
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":USER_PROMPT_TEMPLATE.format(question=question, packed_context=packed_context)}
    ]
    resp = call_llm(messages)
    return {
        "answer": resp,
        "category_router_mode": CATEGORY_ROUTER_MODE,
        "category_hint": cat_hint,
        "router_confidence": cat_conf,
        "used": used
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--faiss_path", default="./cuad_faiss.index")
    args = ap.parse_args()
    result = answer(args.q, k=args.k, faiss_path=args.faiss_path)
    ans = result["answer"]
    cites = result["used"]
    print("\n=== ANSWER ===\n", ans)
    print("\n=== SOURCES (top-k used) ===")
    for i, c in enumerate(cites, 1):
        print(f"[{i}] {c['filename']} | {c['category']}")
