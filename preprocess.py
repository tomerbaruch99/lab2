import pandas as pd

def extract_clauses(csv_path="data/CUAD_v1/master_clauses.csv", save_path="data/CUAD_v1/clauses.csv"):
    df = pd.read_csv(csv_path)
    clause_cols = [c for c in df.columns if c.endswith("-Answer")]

    rows = []
    for _, row in df.iterrows():
        for col in clause_cols:
            text = str(row[col]).strip()
            if text and text != "[]":
                rows.append({
                    "filename": row["Filename"],
                    "category": col.replace("-Answer", ""),
                    "clause_text": text
                })

    clauses = pd.DataFrame(rows)
    clauses.to_csv(save_path, index=False)
    print(f"[✓] Saved {save_path} with {len(clauses)} clauses.")

if __name__ == "__main__":
    extract_clauses()
