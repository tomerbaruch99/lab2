import os, time, argparse, json, re
import pandas as pd
from tqdm.auto import tqdm
from rag_service import answer
from config import (
    EVAL_PROVIDER, EVAL_MODEL, EVAL_TEMP,
    OPENAI_API_KEY, GEMINI_API_KEY, COHERE_API_KEY
)

EVALUATION_PROMPT = """### Task Description:
An instruction (might include an input inside it), a response to evaluate, a reference answer that receives a score of 5, and a score rubric representing multiple evaluation criteria are provided.

1. Write specific and constructive feedback that assesses the response’s quality strictly based on the given score rubrics below. If the response is more detailed or lengthy, this is not a disadvantage unless it includes off-topic or irrelevant content.
2. After writing feedback, provide a score between 1 and 5 for each evaluation criterion.
3. After feedback and scores, provide an overall correctness score (Correct or Incorrect) if the response, in the context of a yes/no question, is correct.
4. Format your output as: "Feedback: {{feedback for each criterion}} [SCORE_FACTUALITY] {{score}} [SCORE_RELEVANCE] {{score}} [SCORE_COMPLETENESS] {{score}} [SCORE_CONFIDENCE] {{score}} [CORRECTNESS] {{Correct or Incorrect}}"
5. Please do not add any other opening, closing, or explanations. Include [SCORE_FACTUALITY], [SCORE_RELEVANCE], [SCORE_COMPLETENESS], [SCORE_CONFIDENCE], and [CORRECTNESS] in your output.

### The instruction to evaluate:
{instruction}

### Response to evaluate:
{response}

### Reference Answer (Score 5):
{reference_answer}

### Score Rubrics:
1. **Factuality**: Is the response correct, accurate, and factual based on the reference answer?
   - Score 1: Completely incorrect, inaccurate, and/or not factual.
   - Score 2: Mostly incorrect, inaccurate, and/or not factual.
   - Score 3: Somewhat correct, accurate, and/or factual.
   - Score 4: Mostly correct, accurate, and factual.
   - Score 5: Completely correct, accurate, and factual.

2. **Relevance**: Does the response stay focused on the instruction and provide relevant information without introducing unnecessary or off-topic content?
   - Score 1: Completely irrelevant to the instruction or question.
   - Score 2: Mostly irrelevant with some on-topic information.
   - Score 3: Somewhat relevant but includes some unnecessary information.
   - Score 4: Mostly relevant with little unnecessary information.
   - Score 5: Fully relevant and focused on the instruction.

3. **Completeness**: Does the response thoroughly cover all parts of the question or instruction without omitting important details?
   - Score 1: Completely incomplete, misses all key points.
   - Score 2: Misses most key points, partially complete.
   - Score 3: Addresses some key points but is incomplete in other aspects.
   - Score 4: Addresses most key points with minor omissions.
   - Score 5: Fully complete, addresses all key points directly.

4. **Confidence**: How confident is the response in providing accurate information based on the reference answer?
   - Score 1: Completely unsure or lacking confidence.
   - Score 2: Mostly unsure, indicates low confidence.
   - Score 3: Somewhat confident but lacks strong evidence.
   - Score 4: Mostly confident with some solid backing.
   - Score 5: Completely confident, well-supported by evidence.

### Feedback:"""

def _call_eval_llm(prompt: str) -> str:
    prov = EVAL_PROVIDER.lower()
    if prov == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(EVAL_MODEL)
        r = model.generate_content(prompt, generation_config={"temperature": EVAL_TEMP})
        return (r.text or "").strip()
    elif prov == "openai":
        import openai
        openai.api_key = OPENAI_API_KEY
        r = openai.ChatCompletion.create(
            model=EVAL_MODEL,
            temperature=EVAL_TEMP,
            messages=[{"role":"user","content":prompt}],
        )
        return r.choices[0].message["content"].strip()
    elif prov == "cohere":
        import cohere
        co = cohere.Client(COHERE_API_KEY)
        r = co.generate(model=EVAL_MODEL, prompt=prompt, temperature=EVAL_TEMP)
        return r.generations[0].text.strip()
    else:
        raise ValueError(f"Unsupported EVAL_PROVIDER: {EVAL_PROVIDER}")

def _extract_scores(text: str):
    # Very close to your ToS extractor
    try:
        feedback = text.split("[SCORE_FACTUALITY]")[0].strip()
        factuality = int(text.split("[SCORE_FACTUALITY]")[1].split("[SCORE_RELEVANCE]")[0].strip())
        relevance  = int(text.split("[SCORE_RELEVANCE]")[1].split("[SCORE_COMPLETENESS]")[0].strip())
        completeness = int(text.split("[SCORE_COMPLETENESS]")[1].split("[SCORE_CONFIDENCE]")[0].strip())
        confidence = int(text.split("[SCORE_CONFIDENCE]")[1].split("[CORRECTNESS]")[0].strip())
        correctness = text.split("[CORRECTNESS]")[1].strip()
        return {
            "eval_feedback": feedback,
            "eval_factuality": factuality,
            "eval_relevance": relevance,
            "eval_completeness": completeness,
            "eval_confidence": confidence,
            "eval_correctness": correctness
        }
    except Exception:
        # fallback when format drifts
        return {
            "eval_feedback": text[:500],
            "eval_factuality": None,
            "eval_relevance": None,
            "eval_completeness": None,
            "eval_confidence": None,
            "eval_correctness": None
        }

def main():
    ap = argparse.ArgumentParser(description="LLM rubric evaluation over CUAD RAG answers.")
    ap.add_argument("--test_csv", default="./testsets/casebank.csv")
    ap.add_argument("--out_csv",  default="./evaluation/eval_results_llm_rubric.csv")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--rate_limit_seconds", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = pd.read_csv(args.test_csv)

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        q = str(r["question"])
        gt = str(r["right_answer"])
        gen, used = answer(q, k=args.k)

        prompt = EVALUATION_PROMPT.format(
            instruction=q,
            response=gen,
            reference_answer=gt
        )
        txt = _call_eval_llm(prompt)
        metrics = _extract_scores(txt)

        rows.append({
            "company": r.get("company",""),
            "category": r.get("category",""),
            "question": q,
            "right_answer": gt,
            "generated_answer": gen,
            **metrics
        })
        time.sleep(args.rate_limit_seconds)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote rubric eval to {args.out_csv}")

if __name__ == "__main__":
    main()
