#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUAD data Preparation
===================================
Implements data preparation (ETL) for:
    A) Contract Clause Q&A Assistant
    B) RAG-driven Contract Reviewer

What this script does
---------------------
1) Loads CUAD's master CSV (83 columns) and SQuAD-style JSON (if provided).
2) Infers the 41 categories from CSV columns that represent clause contexts & answers.
3) Flattens the CSV into a long table, to prep it for the RAG model: (filename, category, context, answer, question_template, answer_type).
4) Cleans clause text while preserving redaction markers (***, ___, <omitted>).
5) Optionally parses & chunks full-contract TXT files into paragraph windows with metadata.
6) Splits the contracts into train, dev, and test sets.
7) (optional) Converts the JSON into a Q&A flat table for fine-tuning.
8) Outputs files (like CSV, Parquet, or JSONL) for downstream retrieval & training.

Usage
-----
python data_preparation.py \
  --csv /path/to/master_clauses.csv \
  --squad_json /path/to/CUAD_v1.json \
  --txt_dir /path/to/full_contract_txt \
  --out_dir ./data_prepared \
  --chunk_chars 4000 \
  --chunk_overlap 400 \
  --val_size 0.15 \
  --test_size 0.15

Notes
-----
- If some inputs are missing (e.g., TXT or JSON), the script will skip those steps and continue.
- The script is redaction-aware: it will not attempt to reconstruct redacted content.
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 41 CUAD categories and answer types (from CUAD README)
# Answer types are approximate and used for normalization/validation.
CUAD_CATEGORIES: List[Dict] = [
    {"name": "Document Name", "answer_type": "text", "group": None},
    {"name": "Parties", "answer_type": "entity_list", "group": None},
    {"name": "Agreement Date", "answer_type": "date", "group": 1},
    {"name": "Effective Date", "answer_type": "date", "group": 1},
    {"name": "Expiration Date", "answer_type": "date_or_perpetual", "group": 1},
    {"name": "Renewal Term", "answer_type": "duration_or_perpetual", "group": 1},
    {"name": "Notice Period To Terminate Renewal", "answer_type": "duration", "group": 1},
    {"name": "Governing Law", "answer_type": "jurisdiction", "group": None},
    {"name": "Most Favored Nation", "answer_type": "yesno", "group": None},
    {"name": "Non-Compete", "answer_type": "yesno", "group": 2},
    {"name": "Exclusivity", "answer_type": "yesno", "group": 2},
    {"name": "No-Solicit Of Customers", "answer_type": "yesno", "group": 2},
    {"name": "Competitive Restriction Exception", "answer_type": "yesno", "group": 2},
    {"name": "No-Solicit Of Employees", "answer_type": "yesno", "group": None},
    {"name": "Non-Disparagement", "answer_type": "yesno", "group": None},
    {"name": "Termination For Convenience", "answer_type": "yesno", "group": None},
    {"name": "Rofr/Rofo/Rofn", "answer_type": "yesno", "group": None},
    {"name": "Change Of Control", "answer_type": "yesno", "group": 3},
    {"name": "Anti-Assignment", "answer_type": "yesno", "group": 3},
    {"name": "Revenue/Profit Sharing", "answer_type": "yesno", "group": None},
    {"name": "Price Restrictions", "answer_type": "yesno", "group": None},
    {"name": "Minimum Commitment", "answer_type": "yesno", "group": None},
    {"name": "Volume Restriction", "answer_type": "yesno", "group": None},
    {"name": "Ip Ownership Assignment", "answer_type": "yesno", "group": None},
    {"name": "Joint Ip Ownership", "answer_type": "yesno", "group": None},
    {"name": "License Grant", "answer_type": "yesno", "group": 4},
    {"name": "Non-Transferable License", "answer_type": "yesno", "group": 4},
    {"name": "Affiliate License-Licensor", "answer_type": "yesno", "group": 4},
    {"name": "Affiliate License-Licensee", "answer_type": "yesno", "group": 4},
    {"name": "Unlimited/All-You-Can-Eat-License", "answer_type": "yesno", "group": None},
    {"name": "Irrevocable Or Perpetual License", "answer_type": "yesno", "group": 4},
    {"name": "Source Code Escrow", "answer_type": "yesno", "group": None},
    {"name": "Post-Termination Services", "answer_type": "yesno", "group": None},
    {"name": "Audit Rights", "answer_type": "yesno", "group": None},
    {"name": "Uncapped Liability", "answer_type": "yesno", "group": 5},
    {"name": "Cap On Liability", "answer_type": "yesno", "group": 5},
    {"name": "Liquidated Damages", "answer_type": "yesno", "group": None},
    {"name": "Warranty Duration", "answer_type": "duration", "group": None},
    {"name": "Insurance", "answer_type": "yesno", "group": None},
    {"name": "Covenant Not To Sue", "answer_type": "yesno", "group": None},
    {"name": "Third Party Beneficiary", "answer_type": "yesno", "group": None},
]

# Natural-language question templates per category (used for RAG prompts)
QUESTION_TEMPLATES: Dict[str, str] = {
    "Document Name": "What is the name of this agreement?",
    "Parties": "Who are the parties to this agreement?",
    "Agreement Date": "On what date was the agreement executed?",
    "Effective Date": "When does the agreement become effective?",
    "Expiration Date": "When does the agreement's initial term expire?",
    "Renewal Term": "What is the renewal term after the initial term expires?",
    "Notice Period To Terminate Renewal": "What notice is required to terminate the renewal?",
    "Governing Law": "Which jurisdiction's law governs this agreement?",
    "Most Favored Nation": "Is there a most-favored-nation clause?",
    "Non-Compete": "Is there a non-compete restriction?",
    "Exclusivity": "Is there an exclusivity obligation?",
    "No-Solicit Of Customers": "Is there a restriction on soliciting customers?",
    "Competitive Restriction Exception": "Are there exceptions to competitive restrictions?",
    "No-Solicit Of Employees": "Is there a restriction on soliciting or hiring employees?",
    "Non-Disparagement": "Is there a non-disparagement requirement?",
    "Termination For Convenience": "Can the agreement be terminated without cause?",
    "Rofr/Rofo/Rofn": "Is there a right of first refusal/offer/negotiation?",
    "Change Of Control": "Are there provisions triggered by a change of control?",
    "Anti-Assignment": "Is consent or notice required to assign the agreement?",
    "Revenue/Profit Sharing": "Is there revenue or profit sharing?",
    "Price Restrictions": "Are there restrictions on pricing changes?",
    "Minimum Commitment": "Is there a minimum purchase or commitment?",
    "Volume Restriction": "Are there volume thresholds with constraints or fees?",
    "Ip Ownership Assignment": "Is IP assigned to the counterparty under any conditions?",
    "Joint Ip Ownership": "Is any IP jointly owned?",
    "License Grant": "Does the agreement grant a license?",
    "Non-Transferable License": "Is the license non-transferable?",
    "Affiliate License-Licensor": "Does the license include the licensor's affiliates or their IP?",
    "Affiliate License-Licensee": "Does the license extend to the licensee's affiliates?",
    "Unlimited/All-You-Can-Eat-License": "Is there an unlimited or enterprise-wide license?",
    "Irrevocable Or Perpetual License": "Is the license irrevocable or perpetual?",
    "Source Code Escrow": "Is source code escrow required?",
    "Post-Termination Services": "Are there post-termination service obligations?",
    "Audit Rights": "Are there audit rights?",
    "Uncapped Liability": "Is liability uncapped for any breach?",
    "Cap On Liability": "Is there a cap on liability?",
    "Liquidated Damages": "Are there liquidated damages or termination fees?",
    "Warranty Duration": "What is the duration of the warranty?",
    "Insurance": "Is insurance required?",
    "Covenant Not To Sue": "Is there a covenant not to sue?",
    "Third Party Beneficiary": "Are there third-party beneficiaries?",
}

# -----------------------------
# Utilities
# -----------------------------

RE_WHITESPACE = re.compile(r"[ \t\f\v]+")
RE_MULTINEWLINES = re.compile(r"\n{3,}")
RE_PAGE_FOOTER = re.compile(r"(?i)page\s+\d+(\s+of\s+\d+)?")
RE_HEADER_FOOTER_NOISE = re.compile(r"(?i)(confidential|treatment request|exhibit|table of contents)")
RE_RED_ACTION = re.compile(r"(\*{2,}|_{2,}|<omitted>)")

def clean_clause_text(text: str) -> str:
    """Clean clause text but preserve redaction markers and <omitted> tokens."""
    if pd.isna(text):
        return ""
    t = str(text)
    # Keep explicit redaction markers as-is; remove obvious repeated whitespace noise
    t = t.replace("\r", "\n")
    t = RE_WHITESPACE.sub(" ", t)
    t = RE_MULTINEWLINES.sub("\n\n", t)
    # Leave redaction markers untouched; do not attempt reconstruction
    return t.strip()

def safe_yesno(x: str) -> Optional[str]:
    if pd.isna(x) or x is None:
        return None
    s = str(x).strip().lower()
    if s in {"yes", "no"}:
        return s.capitalize()
    return None  # leave as None if not clean yes/no

def normalize_answer(category: str, answer: str) -> str:
    """Light normalization for answers; do not infer redactions."""
    if pd.isna(answer) or answer is None:
        return ""
    a = str(answer).strip()
    # Preserve redactions like '1/[]/2014' or '***'
    if RE_RED_ACTION.search(a):
        return a
    # Normalize yes/no
    if category in {c["name"] for c in CUAD_CATEGORIES if c["answer_type"] == "yesno"}:
        yn = safe_yesno(a)
        return yn if yn else a
    # Normalize whitespace
    a = RE_WHITESPACE.sub(" ", a).strip()
    return a

def infer_contract_type(filename: str, document_name_answer: str) -> str:
    """Heuristic: try to infer contract type from Document Name-Answer or filename."""
    candidates = []
    if document_name_answer and isinstance(document_name_answer, str):
        candidates.append(document_name_answer.lower())
    if filename:
        candidates.append(Path(filename).stem.lower())
    joined = " ".join(candidates)
    # Common CUAD contract types keywords
    types = [
        "affiliate agreement","agency agreement","collaboration","cooperation","co-branding",
        "consulting","development","distributor","endorsement","franchise","hosting","ip agreement",
        "joint venture","license agreement","maintenance","manufacturing","marketing",
        "non-compete","no-solicit","non-disparagement","outsourcing","promotion","reseller",
        "service agreement","sponsorship","supply agreement","strategic alliance","transportation"
    ]
    for t in types:
        if t in joined:
            return t
    return "unknown"

def chunk_text(text: str, max_chars: int = 4000, overlap: int = 400) -> List[Tuple[int, str]]:
    """Chunk plain text by characters with overlap; page/section aware chunking can be added later."""
    text = clean_clause_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        # try to end at a sentence boundary
        cut = text.rfind(". ", start, end)
        if cut == -1 or cut <= start + 200:
            cut = end
        else:
            cut += 1  # keep the period
        chunks.append((start, text[start:cut].strip()))
        if cut >= n:
            break
        start = max(0, cut - overlap)
    return chunks

def ensure_out_dir(d: str) -> Path:
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p

# -----------------------------
# Stage 1: ETL Pipeline
# -----------------------------

def read_master_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names (strip BOMs/whitespace)
    df.columns = [c.strip() for c in df.columns]
    return df

def infer_categories_from_columns(columns: List[str]) -> List[Tuple[str, str]]:
    """
    Return list of (context_col, answer_col) pairs in the order they appear,
    excluding the leading 'Filename' column.
    """
    pairs = []
    for c in columns:
        if c == "Filename":
            continue
        if c.endswith("-Answer"):
            # handled when we see the context name
            continue
        context_col = c
        answer_col = f"{c}-Answer"
        if answer_col in columns:
            pairs.append((context_col, answer_col))
        else:
            # Some CSVs may have slight colcase diffs; try case-insensitive match
            lower_map = {x.lower(): x for x in columns}
            key = (context_col + "-Answer").lower()
            if key in lower_map:
                pairs.append((context_col, lower_map[key]))
            else:
                # Keep context without answer if truly missing
                pairs.append((context_col, None))
    return pairs

def build_long_table(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten wide 83-col master into long (filename, category, context, answer, ...)"""
    pairs = infer_categories_from_columns(df.columns.tolist())
    rows = []
    for _, row in df.iterrows():
        filename = row.get("Filename", "")
        docname_ans = row.get("Document Name-Answer", "") or ""
        contract_type = infer_contract_type(filename, docname_ans)
        for context_col, answer_col in pairs:
            category = context_col.strip()
            if category == "Filename":
                continue
            context = clean_clause_text(row.get(context_col, ""))
            answer_raw = row.get(answer_col, "") if answer_col else ""
            answer = normalize_answer(category, answer_raw)
            answer_type = next((c["answer_type"] for c in CUAD_CATEGORIES if c["name"] == category), "text")
            question_template = QUESTION_TEMPLATES.get(category, f"What is the answer for category '{category}'?")
            rows.append({
                "filename": filename,
                "contract_type": contract_type,
                "category": category,
                "context": context,
                "answer": answer,
                "answer_type": answer_type,
                "question_template": question_template
            })
    long_df = pd.DataFrame(rows)
    # Drop empty contexts to keep the retriever clean
    long_df = long_df[long_df["context"].astype(str).str.strip() != ""]
    long_df.reset_index(drop=True, inplace=True)
    return long_df

def build_contract_paragraph_index(txt_dir: str,
                                   out_path: str,
                                   chunk_chars: int = 4000,
                                   overlap: int = 400) -> pd.DataFrame:
    """
    Chunk all TXT contracts; return/save a dataframe:
    (doc_id, filename, chunk_id, start_char, text)
    """
    records = []
    txt_dir_p = Path(txt_dir)
    if not txt_dir_p.exists():
        print(f"[WARN] TXT folder not found, skipping: {txt_dir}")
        return pd.DataFrame()

    for p in sorted(txt_dir_p.glob("*.txt")):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = p.read_text(encoding="latin-1", errors="ignore")
        chunks = chunk_text(text, max_chars=chunk_chars, overlap=overlap)
        for i, (start, chunk) in enumerate(chunks):
            records.append({
                "doc_id": p.stem,
                "filename": p.name,
                "chunk_id": i,
                "start_char": start,
                "text": chunk
            })
    para_df = pd.DataFrame(records)
    if not para_df.empty:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        if outp.suffix.lower() == ".parquet":
            para_df.to_parquet(outp, index=False)
        else:
            para_df.to_csv(outp, index=False)
        print(f"[OK] Paragraph index saved to: {outp}")
    else:
        print("[INFO] No TXT files processed.")
    return para_df

def make_splits(contract_df: pd.DataFrame,
                val_size: float = 0.15,
                test_size: float = 0.15,
                seed: int = 42) -> pd.DataFrame:
    """
    Make contract-level splits by filename, stratified by contract_type where possible.
    Returns a mapping dataframe with columns: filename, split.
    """
    rng = np.random.RandomState(seed)
    # Unique contracts
    meta = contract_df[["filename", "contract_type"]].drop_duplicates().copy()
    # Stratify by contract_type if there are enough per class; else random
    def stratified_split(group: pd.DataFrame) -> pd.DataFrame:
        n = len(group)
        idx = list(group.index)
        rng.shuffle(idx)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        test_idx = set(idx[:n_test])
        val_idx = set(idx[n_test:n_test+n_val])
        split = []
        for i in group.index:
            if i in test_idx:
                split.append("test")
            elif i in val_idx:
                split.append("val")
            else:
                split.append("train")
        out = group.copy()
        out["split"] = split
        return out

    # apply per contract_type
    parts = []
    for ct, sub in meta.groupby("contract_type"):
        parts.append(stratified_split(sub))
    split_map = pd.concat(parts, ignore_index=True)
    return split_map

def attach_splits(long_df: pd.DataFrame, split_map: pd.DataFrame) -> pd.DataFrame:
    out = long_df.merge(split_map, on=["filename", "contract_type"], how="left")
    # Default to train if missing
    out["split"] = out["split"].fillna("train")
    return out

def load_squad_json(squad_path: str) -> pd.DataFrame:
    """
    Load SQuAD-style CUAD JSON and flatten to:
    (filename, paragraph_id, category, question, answer_text, answer_start, is_impossible, context)
    """
    with open(squad_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for article in data.get("data", []):
        title = article.get("title", "")
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                qid = qa.get("id", "")
                question = qa.get("question", "")
                is_impossible = qa.get("is_impossible", False)
                answers = qa.get("answers", [])
                if answers:
                    for ans in answers:
                        rows.append({
                            "filename": title,
                            "paragraph_id": hash(context) & 0xffffffff,
                            "question_id": qid,
                            "question": question,
                            "answer_text": ans.get("text", ""),
                            "answer_start": ans.get("answer_start", -1),
                            "is_impossible": is_impossible,
                            "context": context
                        })
                else:
                    rows.append({
                        "filename": title,
                        "paragraph_id": hash(context) & 0xffffffff,
                        "question_id": qid,
                        "question": question,
                        "answer_text": "",
                        "answer_start": -1,
                        "is_impossible": is_impossible,
                        "context": context
                    })
    return pd.DataFrame(rows)

def write_outputs(out_dir: str,
                  long_df: pd.DataFrame,
                  squad_df: Optional[pd.DataFrame],
                  para_df: Optional[pd.DataFrame]):
    out_dir_p = ensure_out_dir(out_dir)
    # Long, RAG-ready clauses
    long_path = out_dir_p / "cuad_long_clauses.parquet"
    long_df.to_parquet(long_path, index=False)
    # Also save a compact JSONL for retrieval systems
    jsonl_path = out_dir_p / "cuad_long_clauses.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, r in long_df.iterrows():
            f.write(json.dumps({
                "id": f"{r.filename}|{r.category}",
                "text": r.context,
                "metadata": {
                    "filename": r.filename,
                    "contract_type": r.contract_type,
                    "category": r.category,
                    "answer_type": r.answer_type,
                    "question_template": r.question_template,
                    "answer": r.answer
                }
            }, ensure_ascii=False) + "\n")
    print(f"[OK] RAG-ready clauses: {long_path} ; {jsonl_path}")

    # SQuAD flat (optional)
    if squad_df is not None and not squad_df.empty:
        squad_path = out_dir_p / "cuad_squad_flat.parquet"
        squad_df.to_parquet(squad_path, index=False)
        print(f"[OK] SQuAD-flat saved: {squad_path}")

    # Paragraph index (optional)
    if para_df is not None and not para_df.empty:
        para_path = out_dir_p / "cuad_paragraph_index.parquet"
        para_df.to_parquet(para_path, index=False)
        print(f"[OK] Paragraph index saved: {para_path}")

def main():
    parser = argparse.ArgumentParser(description="CUAD RAG Preparation (Stages 0 & 1)")
    parser.add_argument("--csv", type=str, required=True, help="Path to CUAD_v1 master CSV (83 columns)")
    parser.add_argument("--squad_json", type=str, default=None, help="Path to SQuAD-style CUAD JSON (optional)")
    parser.add_argument("--txt_dir", type=str, default=None, help="Folder with full_contracts_txt (optional)")
    parser.add_argument("--out_dir", type=str, default="./cuad_prepared")
    parser.add_argument("--chunk_chars", type=int, default=4000, help="Max characters per TXT chunk")
    parser.add_argument("--chunk_overlap", type=int, default=400, help="Character overlap between chunks")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation split ratio (by contract)")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test split ratio (by contract)")
    args = parser.parse_args()

    # 1) Read master CSV and build long table
    print("[STEP] Loading master CSV...")
    df = read_master_csv(args.csv)

    # Validate expected columns
    expected_first = "Filename"
    if expected_first not in df.columns:
        raise ValueError(f"CSV missing '{expected_first}' column. Found: {df.columns[:10].tolist()}")

    print("[STEP] Building long, RAG-ready table...")
    long_df = build_long_table(df)

    # 2) Build contract-level splits
    print("[STEP] Making contract-level splits...")
    split_map = make_splits(long_df, val_size=args.val_size, test_size=args.test_size, seed=42)
    long_df = attach_splits(long_df, split_map)

    # 3) Optional: Chunk full TXT contracts into paragraph windows
    para_df = pd.DataFrame()
    if args.txt_dir:
        print("[STEP] Chunking full-contract TXT files...")
        para_df = build_contract_paragraph_index(
            txt_dir=args.txt_dir,
            out_path=os.path.join(args.out_dir, "cuad_paragraph_index.csv"),
            chunk_chars=args.chunk_chars,
            overlap=args.chunk_overlap,
        )

    # 4) Optional: Flatten SQuAD JSON for extractive reader fine-tuning
    squad_df = pd.DataFrame()
    if args.squad_json and Path(args.squad_json).exists():
        print("[STEP] Loading SQuAD-style JSON...")
        squad_df = load_squad_json(args.squad_json)

    # 5) Write outputs
    print("[STEP] Writing outputs...")
    write_outputs(args.out_dir, long_df, squad_df, para_df)

    # 6) Quick report
    print("\n[REPORT]")
    print(f" Clauses rows: {len(long_df):,} (train={sum(long_df.split=='train'):,}, val={sum(long_df.split=='val'):,}, test={sum(long_df.split=='test'):,})")
    print(f" Unique contracts: {long_df['filename'].nunique():,}")
    print(f" Categories covered: {long_df['category'].nunique()}")
    if not para_df.empty:
        print(f" Paragraph chunks: {len(para_df):,}")
    if not squad_df.empty:
        print(f" SQuAD QA rows: {len(squad_df):,}")

if __name__ == "__main__":
    main()
