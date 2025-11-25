"""
Builds a labeled QA test set from CUAD's prepared clauses.

Output: eval_data/testset.xlsx with columns:
- Company        (Holds the contract filename)
- Question
- Right Answer
- Context        (Clause text used as ground truth support)
- Category
- ContractType
- Split
"""

from pathlib import Path

import numpy as np
import pandas as pd

CUAD_PREPARED_DIR = "./cuad_prepared_data"
CUAD_LONG_CLAUSES = "cuad_long_clauses.parquet"
OUT_DIR = Path("eval_data")
OUT_PARQUET = OUT_DIR / "testset.parquet"
OUT_CSV = OUT_DIR / "testset.csv"

# How many QA items to sample in total
N_SAMPLES = 200
RANDOM_SEED = 42


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clauses_path = Path(CUAD_PREPARED_DIR) / CUAD_LONG_CLAUSES
    if not clauses_path.exists():
        raise FileNotFoundError(f"Cannot find {clauses_path}. Run data_preparation.py first.")

    df = pd.read_parquet(clauses_path)

    # Keep only rows with non-empty answer and question_template
    df = df.copy()
    df["answer"] = df["answer"].astype(str).str.strip()
    df["question_template"] = df["question_template"].astype(str).str.strip()
    df["context"] = df["context"].astype(str).str.strip()

    df = df[
        (df["answer"] != "") &
        (df["question_template"] != "") &
        (df["context"] != "")
    ]

    # Prefer using the test split for evaluation, if it exists
    if "split" in df.columns:
        test_df = df[df["split"] == "test"].copy()
        if len(test_df) >= 50:  # sanity check
            df = test_df

    if len(df) <= N_SAMPLES:
        sampled = df
    else:
        # Stratified-ish sampling across categories
        rng = np.random.RandomState(RANDOM_SEED)
        categories = df["category"].unique().tolist()
        per_cat = max(1, N_SAMPLES // max(1, len(categories)))
        pieces = []
        for cat in categories:
            sub = df[df["category"] == cat]
            n = min(len(sub), per_cat)
            if n > 0:
                pieces.append(sub.sample(n=n, random_state=rng))
        sampled = pd.concat(pieces, ignore_index=True)
        # If still short, top up randomly
        if len(sampled) < N_SAMPLES and len(df) > len(sampled):
            remaining = df.drop(sampled.index)
            extra_n = min(N_SAMPLES - len(sampled), len(remaining))
            sampled = pd.concat(
                [sampled, remaining.sample(n=extra_n, random_state=rng)],
                ignore_index=True
            )

    # Build testset in the same style as ToS project
    testset = pd.DataFrame({
        # We reuse "Company" column name to stay compatible with your existing evaluation code.
        # Here it represents the contract filename.
        "Company": sampled["filename"],
        "Question": sampled["question_template"],
        "Right Answer": sampled["answer"],
        "Context": sampled["context"],
        "Category": sampled["category"],
        "ContractType": sampled["contract_type"],
        "Split": sampled.get("split", "unknown"),
    })

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    testset.to_parquet(OUT_PARQUET, index=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    testset.to_csv(OUT_CSV, index=False)
    print(f"Saved testset to: {OUT_PARQUET} and {OUT_CSV} with {len(testset)} rows")

if __name__ == "__main__":
    main()
