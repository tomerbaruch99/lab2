import os, json, time, random, argparse
from typing import List, Dict
import pandas as pd

from consts import load_experiment_config
config = load_experiment_config()

RANDOM_SEED = 48
random.seed(RANDOM_SEED)

# ---------- Load CUAD long table ----------
def load_long() -> pd.DataFrame:
    if os.path.exists(LONG_PARQUET):
        df = pd.read_parquet(LONG_PARQUET)
    elif os.path.exists(LONG_JSONL):
        rows=[]
        with open(LONG_JSONL,"r",encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                m = o["metadata"]
                rows.append({
                    "filename": m.get("filename",""),
                    "contract_type": m.get("contract_type","unknown"),
                    "category": m.get("category",""),
                    "question_template": m.get("question_template",""),
                    "answer": m.get("answer",""),
                    "context": o.get("text",""),
                    "split": m.get("split","train"),
                })
        df = pd.DataFrame(rows)
    else:
        raise FileNotFoundError("CUAD outputs not found. Expected parquet or jsonl.")
    # basic hygiene
    df = df.fillna("")
    df["context"] = df["context"].astype(str).str.strip()
    df = df[df["context"] != ""]
    return df

# ---------- LLM clients ----------
def _llm_call(prompt: str) -> str:
    prov = SYNTH_PROVIDER.lower()
    if prov == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(SYNTH_MODEL)
        r = model.generate_content(prompt, generation_config={"temperature": SYNTH_TEMP})
        return (r.text or "").strip()
    elif prov == "openai":
        import openai
        openai.api_key = OPENAI_API_KEY
        r = openai.ChatCompletion.create(
            model=SYNTH_MODEL,
            temperature=SYNTH_TEMP,
            messages=[{"role":"user","content":prompt}],
        )
        return r.choices[0].message["content"].strip()
    elif prov == "cohere":
        import cohere
        co = cohere.Client(COHERE_API_KEY)
        r = co.generate(
            model=SYNTH_MODEL,
            prompt=prompt,
            temperature=SYNTH_TEMP,
        )
        return r.generations[0].text.strip()
    else:
        raise ValueError(f"Unsupported SYNTH_PROVIDER: {SYNTH_PROVIDER}")

# ---------- Prompt ----------
QA_PROMPT = """
You are generating one (1) practical, fact-based Q&A from a legal contract excerpt.

Rules:
- Ask a real user-style **factoid** question about the excerpt, referencing the **filename** as the agreement entity (e.g., “In <filename>, ...?”).
- The answer must be supported strictly by the excerpt; if unclear, say it’s not stated.
- Be concise, precise, and neutral. Preserve any redactions exactly (***, ___, <omitted>).
- Do NOT mention “the passage” or “context”.

Return ONLY this schema (no extra text):
Factoid question: <question>
Answer: <answer>

Filename: {filename}
Category: {category}
Excerpt:
\"\"\"{context}\"\"\"
"""

def _parse_qa(text: str) -> Dict[str,str]:
    # Robust split with fallbacks
    q = ""
    a = ""
    t = text.replace("\r","")
    if "Factoid question:" in t and "Answer:" in t:
        q = t.split("Factoid question:")[-1].split("Answer:")[0].strip()
        a = t.split("Answer:")[-1].strip()
    else:
        # fallback: first line as question, remainder as answer
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        if lines:
            q = lines[0].rstrip("?") + "?"
            a = " ".join(lines[1:]).strip()
    return {"question": q, "answer": a}

# ---------- Builders ----------
def build_grounded(df: pd.DataFrame, n_per_category: int) -> pd.DataFrame:
    df = df[(df["question_template"]!="") & (df["answer"]!="")]
    out = []
    for cat, sub in df.groupby("category"):
        take = sub.sample(n=min(n_per_category, len(sub)), random_state=RANDOM_SEED)
        for _, r in take.iterrows():
            out.append({
                "company": r["filename"],
                "category": r["category"],
                "question": r["question_template"],
                "right_answer": str(r["answer"]).strip(),
                "source": "grounded",
            })
    return pd.DataFrame(out)

def build_synthetic(df: pd.DataFrame, n_per_category: int, rate_limit_s: float) -> pd.DataFrame:
    # sample contexts per category
    out = []
    for cat, sub in df.groupby("category"):
        take = sub.sample(n=min(n_per_category, len(sub)), random_state=RANDOM_SEED)
        for _, r in take.iterrows():
            prompt = QA_PROMPT.format(
                filename=r["filename"] or "the agreement",
                category=r["category"],
                context=r["context"]
            )
            try:
                txt = _llm_call(prompt)
                qa = _parse_qa(txt)
                if qa["question"] and qa["answer"]:
                    out.append({
                        "company": r["filename"],
                        "category": r["category"],
                        "question": qa["question"],
                        "right_answer": qa["answer"],
                        "source": "synthetic",
                    })
            except Exception as e:
                # soft-fail and continue
                pass
            time.sleep(rate_limit_s)
    return pd.DataFrame(out)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Create CUAD test sets (grounded and/or synthetic).")
    ap.add_argument("--mode", choices=["grounded","synthetic","both"], default="grounded",
                    help="grounded=use CUAD QA templates; synthetic=LLM-generated; both=concat.")
    ap.add_argument("--n_per_category", type=int, default=20)
    ap.add_argument("--use_split", default="test", choices=["train","val","test","all"])
    ap.add_argument("--out_csv", default="./testsets/casebank.csv")
    ap.add_argument("--rate_limit_seconds", type=float, default=0.5,
                    help="Delay between synthetic generations to avoid rate limits.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    df = load_long()
    if args.use_split != "all" and "split" in df.columns:
        df = df[df["split"] == args.use_split]

    frames = []
    if args.mode in ("grounded","both"):
        grounded = build_grounded(df, n_per_category=args.n_per_category)
        frames.append(grounded)

    if args.mode in ("synthetic","both"):
        # Cap by env var if user prefers
        n_cat = min(args.n_per_category, SYNTH_MAX_PER_CATEGORY)
        synthetic = build_synthetic(df, n_per_category=n_cat, rate_limit_s=args.rate_limit_seconds)
        frames.append(synthetic)

    if not frames:
        raise ValueError("No data produced. Check mode and inputs.")

    out = pd.concat(frames, ignore_index=True)
    # de-dup identical (company, category, question)
    out = out.drop_duplicates(subset=["company","category","question"]).reset_index(drop=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote {len(out)} rows to {args.out_csv} (mode={args.mode})")

if __name__ == "__main__":
    main()
