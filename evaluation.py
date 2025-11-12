import os, argparse, pandas as pd, json, time
from tqdm.auto import tqdm
from rag import answer
from evaluation_metrics import exactish_match, compute_retrieval_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", default="./testsets/casebank.csv")
    ap.add_argument("--out_csv", default="./evaluation/eval_results.csv")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    df = pd.read_csv(args.test_csv)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        q = r["question"]
        ans, used = answer(q, k=args.k)
        # basic correctness (string containment proxy)
        correctness = exactish_match(ans, r["right_answer"])
        ret = compute_retrieval_metrics(used, r["company"], r["category"])
        rows.append({
            "company": r["company"],
            "category": r["category"],
            "question": q,
            "right_answer": r["right_answer"],
            "generated_answer": ans,
            "correctness_strict": correctness,
            "hit_filename": ret["hit_filename"],
            "hit_category": ret["hit_category"]
        })
        time.sleep(0.1)  # be gentle on API

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved eval to {args.out_csv}")

if __name__ == "__main__":
    main()
