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
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from consts import CUAD_CSV_FILEPATH, CUAD_JSON_FILEPATH, FULL_CONTRACTS_TXT_DIR

# 41 CUAD categories and answer types (taken from CUAD README)
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

# Natural-language question templates per category
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

# CUAD contract types (from CUAD_v1_README.txt)
CONTRACT_TYPES: List[str] = [
    "affiliate agreement",
    "agency agreement",
    "collaboration/cooperation agreement",
    "collaboration agreement",
    "cooperation agreement",
    "co-branding agreement",
    "consulting agreement",
    "development agreement",
    "distributor agreement",
    "endorsement agreement",
    "franchise agreement",
    "hosting agreement",
    "ip agreement",
    "joint venture agreement",
    "license agreement",
    "maintenance agreement",
    "manufacturing agreement",
    "marketing agreement",
    "non-compete/no-solicit/non-disparagement agreement",
    "non-compete agreement",
    "no-solicit agreement",
    "non-disparagement agreement",
    "outsourcing agreement",
    "promotion agreement",
    "reseller agreement",
    "service agreement",
    "sponsorship agreement",
    "supply agreement",
    "strategic alliance agreement",
    "transportation agreement",
]

# -----------------------------
# Utilities

RE_WHITESPACE = re.compile(r"[ \t\f\v]+")  # Whitespace like tabs, spaces; doesn't include newlines.
RE_PAGE_FOOTER = re.compile(r"(?i)page\s+\d+(\s+of\s+\d+)?")  # Page numbers like "page 1" or "page 1 of 10".
RE_HEADER_FOOTER_NOISE = re.compile(r"(?i)(confidential|treatment request|exhibit|table of contents)")  # Header/footer noise like "Confidential", "Treatment Request", "Exhibit", "Table of Contents".
RE_RED_ACTION = re.compile(r"\*{2,}|_{2,}|<omitted>")  # Redaction markers like "**", "__", "<omitted>".

def clean_clause_text(text: str) -> str:
    """Cleans the input clause text, but keeps redaction markers and <omitted> tokens."""
    # Handle NaN, None, or non-string inputs
    if text is None:
        return ""
    # Check for pandas NaN (which is a float)
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass  # Not a type that can be NaN
    # Convert to string and process
    text = str(text)
    # Also check for string "nan" or "None" that might come from CSV
    if text.lower() in ("nan", "none", ""):
        return ""
    return RE_PAGE_FOOTER.sub("", RE_HEADER_FOOTER_NOISE.sub("", RE_WHITESPACE.sub(" ", text).replace("\r", "\n")))

def normalize_answer(category: str, answer: str) -> str:
    """Light normalization for answers; does not infer redactions. Preserves redactions like '1/[]/2014' or '***'."""
    # clean_clause_text handles NaN/None, so we can call it directly
    answer = clean_clause_text(answer)
    if not answer:
        return ""
    if RE_RED_ACTION.search(answer):
        return answer
    if category in {c["name"] for c in CUAD_CATEGORIES if c["answer_type"] == "yesno"}:
        return answer.strip().capitalize() if answer.strip().lower() in ["yes", "no"] else answer
    return answer.strip()

def infer_contract_type(filename: str, document_name_answer: str) -> str:
    """Heuristic: try to infer contract type from Document Name-Answer or filename. Returns 'unknown' if no match."""
    candidates: List[str] = []
    if document_name_answer:
        candidates.append(document_name_answer.lower())
    if filename:
        candidates.append(os.path.splitext(filename)[0].lower())
    joined_candidates = " ".join(candidates).lower()
    for contract_type in CONTRACT_TYPES:
        if contract_type in joined_candidates:
            return contract_type
    return "unknown"

def chunk_text(text: str, max_chars: int = 4000, overlap: int = 400) -> List[Tuple[int, str]]:
    """Chunk plain text by characters with overlap; page/section aware chunking can be added later."""
    if not text or pd.isna(text):
        return []
    text = clean_clause_text(str(text))
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    # Calculate reasonable max iterations: worst case is when we advance by 1 char each time
    # But with overlap, we should advance by at least (max_chars - overlap) each iteration
    # Add generous buffer (3x) to account for edge cases
    min_advance = max(1, max_chars - overlap)
    max_iterations = (n // min_advance) * 3 + 100  # Much more generous limit
    iterations = 0
    
    while start < n:
        iterations += 1
        if iterations > max_iterations:
            # Safety: if we've iterated too many times, break to prevent infinite loop
            # This is a fallback safety mechanism
            print(f"[WARN] chunk_text: Max iterations ({max_iterations}) reached for text of length {n}, breaking at start={start}")
            break
            
        end = min(start + max_chars, n)
        # try to end at a sentence boundary
        cut = text.rfind(". ", start, end)
        if cut == -1 or cut <= start + 200:
            cut = end
        else:
            cut += 1  # keep the period
        
        chunk_text_slice = text[start:cut].strip()
        if chunk_text_slice:  # Only add non-empty chunks
            chunks.append((start, chunk_text_slice))
        
        if cut >= n:
            break
        
        # Calculate next start position
        next_start = max(0, cut - overlap)
        
        # Safety check: ensure we're making progress
        if next_start <= start:
            # If we're not making progress, force advancement
            next_start = start + 1
            if next_start >= n:
                break
        
        start = next_start
    
    return chunks

def ensure_out_dir(d: str) -> Path:
    """Create output directory if it doesn't exist and return Path object."""
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p

# -----------------------------
# Stage 1: ETL Pipeline

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
                                   overlap: int = 400,
                                   batch_size: int = 50) -> pd.DataFrame:
    """
    Chunk all TXT contracts; return/save a dataframe:
    (doc_id, filename, chunk_id, start_char, text)
    Processes files in batches and writes incrementally to reduce memory usage.
    """
    txt_dir_p = Path(txt_dir)
    if not txt_dir_p.exists():
        print(f"[WARN] TXT folder not found, skipping: {txt_dir}")
        return pd.DataFrame()

    # Get file list (don't sort to save memory - process as we find them)
    txt_files = list(txt_dir_p.glob("*.txt"))
    total_files = len(txt_files)
    if total_files == 0:
        print("[INFO] No TXT files found.")
        return pd.DataFrame()

    print(f"[INFO] Found {total_files} TXT files. Processing and writing after each file...")
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    
    # Use CSV for incremental writing, then convert to Parquet if needed
    is_parquet = outp.suffix.lower() == ".parquet"
    temp_csv = outp.with_suffix(".csv.tmp") if is_parquet else None
    write_path = temp_csv if is_parquet else outp
    
    # Remove existing file if it exists to start fresh
    if write_path.exists():
        write_path.unlink()
    
    processed = 0
    skipped = 0
    total_chunks = 0
    header_written = False
    
    print(f"[INFO] Starting file processing...")
    
    try:
        for file_idx, p in enumerate(txt_files, 1):
            try:
                # Read file with size limit check (skip large files)
                file_size = p.stat().st_size
                if file_size > 10 * 1024 * 1024:  # Skip files larger than 10MB
                    skipped += 1
                    if skipped <= 5:  # Only print first 5 skipped files
                        print(f"[WARN] Skipping large file: {p.name} ({file_size / 1024 / 1024:.1f}MB)")
                    continue
                if file_size == 0:
                    skipped += 1
                    continue
                
                # Read file
                text = None
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    try:
                        text = p.read_text(encoding="latin-1", errors="ignore")
                    except Exception as e2:
                        print(f"[WARN] Could not read {p.name}: {e2}")
                        continue
                
                if not text:
                    continue
                
                # Process chunks
                try:
                    chunks = chunk_text(text, max_chars=chunk_chars, overlap=overlap)
                except Exception as e:
                    print(f"[ERROR] Error chunking {p.name}: {e}")
                    del text
                    continue
                finally:
                    del text  # Free memory immediately after chunking
                
                if not chunks:
                    continue
                
                # Write chunks directly to CSV using built-in csv module (more memory-efficient)
                try:
                    num_chunks = len(chunks)
                    
                    # Open file in append mode
                    with open(write_path, 'a', newline='', encoding='utf-8') as csvfile:
                        # Use QUOTE_ALL to quote all fields - safest for text with commas, newlines, quotes
                        # This ensures proper CSV formatting even with complex text content
                        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                        # Write header if first file
                        if not header_written:
                            writer.writerow(['doc_id', 'filename', 'chunk_id', 'start_char', 'text'])
                            header_written = True
                        
                        # Write each chunk directly (no DataFrame creation)
                        for i, (start_pos, chunk) in enumerate(chunks):
                            writer.writerow([p.stem, p.name, i, start_pos, chunk])
                    
                    processed += 1
                    total_chunks += num_chunks
                except Exception as e:
                    print(f"[ERROR] Error writing chunks for {p.name}: {e}")
                    processed += 1
                finally:
                    del chunks  # Free memory immediately
                
                # Progress update more frequently
                if processed % 10 == 0 or file_idx == total_files:
                    print(f"[INFO] Processed {processed}/{total_files} files ({total_chunks:,} chunks so far)...")
                    
            except MemoryError:
                print(f"[ERROR] Out of memory processing {p.name}, skipping...")
                continue
            except Exception as e:
                print(f"[WARN] Error processing {p.name}: {e}")
                continue
        
        # Convert CSV to Parquet if needed (using pyarrow for efficient conversion)
        if is_parquet and temp_csv and temp_csv.exists():
            csv_size_mb = temp_csv.stat().st_size / (1024 * 1024)
            print(f"[INFO] Converting CSV to Parquet ({csv_size_mb:.1f}MB, this may take a moment)...")
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                import pyarrow.csv as pc
                
                # Use pyarrow to convert CSV to Parquet efficiently
                # Note: This may load the entire CSV into memory, but PyArrow is more efficient than pandas
                parse_opts = pc.ParseOptions(newlines_in_values=True)
                table = pc.read_csv(temp_csv, parse_options=parse_opts)
                pq.write_table(table, outp)
                row_count = table.num_rows
                del table  # Free memory
                temp_csv.unlink()  # Delete temporary CSV
                print(f"[OK] Paragraph index saved to: {outp} ({row_count:,} chunks from {processed} files)")
                # Return empty DataFrame with schema (data is on disk, no need to load)
                return pd.DataFrame(columns=['doc_id', 'filename', 'chunk_id', 'start_char', 'text'])
            except ImportError:
                # Fallback to pandas if pyarrow not available (less efficient)
                print("[WARN] PyArrow not available, using pandas (slower, may use more memory)...")
                # Read in chunks and write incrementally
                chunk_list = []
                for chunk in pd.read_csv(temp_csv, chunksize=50000):
                    chunk_list.append(chunk)
                    if len(chunk_list) >= 2:  # Combine every 2 chunks to reduce memory
                        combined = pd.concat(chunk_list, ignore_index=True)
                        if not outp.exists():
                            combined.to_parquet(outp, index=False)
                        else:
                            existing = pd.read_parquet(outp)
                            pd.concat([existing, combined], ignore_index=True).to_parquet(outp, index=False)
                            del existing
                        chunk_list = []
                        del combined
                # Handle remaining chunks
                if chunk_list:
                    combined = pd.concat(chunk_list, ignore_index=True)
                    if not outp.exists():
                        combined.to_parquet(outp, index=False)
                    else:
                        existing = pd.read_parquet(outp)
                        pd.concat([existing, combined], ignore_index=True).to_parquet(outp, index=False)
                temp_csv.unlink()
                print(f"[OK] Paragraph index saved to: {outp} (from {processed} files)")
                return pd.DataFrame(columns=['doc_id', 'filename', 'chunk_id', 'start_char', 'text'])
            except MemoryError:
                print(f"[WARN] Out of memory converting to Parquet. Keeping CSV file: {temp_csv}")
                print(f"[INFO] You can manually convert {temp_csv} to Parquet later if needed.")
                return pd.DataFrame(columns=['doc_id', 'filename', 'chunk_id', 'start_char', 'text'])
            except Exception as e:
                print(f"[WARN] Error converting to Parquet: {e}. Keeping CSV file: {temp_csv}")
                return pd.DataFrame(columns=['doc_id', 'filename', 'chunk_id', 'start_char', 'text'])
        elif write_path.exists():
            # For CSV, return empty DataFrame with schema (data already on disk)
            print(f"[OK] Paragraph index saved to: {write_path} ({total_chunks:,} chunks from {processed} files)")
            # Return minimal metadata
            return pd.DataFrame(columns=['doc_id', 'filename', 'chunk_id', 'start_char', 'text'])
        else:
            print("[INFO] No TXT files processed.")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"[ERROR] Failed to process files: {e}")
        # Clean up temp file on error
        if temp_csv and temp_csv.exists():
            temp_csv.unlink()
        raise

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

    # Paragraph index (optional) - already written by build_contract_paragraph_index
    # Just report if it exists
    para_path = out_dir_p / "cuad_paragraph_index.parquet"
    if para_path.exists():
        print(f"[OK] Paragraph index already saved: {para_path}")
    elif (out_dir_p / "cuad_paragraph_index.csv").exists():
        print(f"[OK] Paragraph index already saved: {out_dir_p / 'cuad_paragraph_index.csv'}")

def main():
    parser = argparse.ArgumentParser(description="CUAD RAG Preparation (Stages 0 & 1)")
    parser.add_argument("--out_dir", type=str, default="./cuad_prepared_data")
    parser.add_argument("--chunk_chars", type=int, default=4000, help="Max characters per TXT chunk")
    parser.add_argument("--chunk_overlap", type=int, default=400, help="Character overlap between chunks")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation split ratio (by contract)")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test split ratio (by contract)")
    args = parser.parse_args()

    # Read master CSV and build long table
    print("[STEP] Loading master CSV...")
    df = read_master_csv(CUAD_CSV_FILEPATH)

    # Validate expected columns
    expected_first = "Filename"
    if expected_first not in df.columns:
        raise ValueError(f"CSV missing '{expected_first}' column. Found: {df.columns[:10].tolist()}")

    print("[STEP] Building long, RAG-ready table...")
    long_df = build_long_table(df)

    # Build contract-level splits
    print("[STEP] Making contract-level splits...")
    split_map = make_splits(long_df, val_size=args.val_size, test_size=args.test_size, seed=42)
    long_df = attach_splits(long_df, split_map)

    # Optional: Chunk full TXT contracts into paragraph windows
    # Free up memory from previous steps before chunking
    import gc
    # Save long_df to disk first before freeing memory
    long_df_path = os.path.join(args.out_dir, "cuad_long_clauses_temp.parquet")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(long_df_path, index=False)
    print(f"[INFO] Saved long_df to disk, freeing memory...")
    
    # Free memory from previous steps
    del df  # Free the original CSV DataFrame
    long_df_size_mb = long_df.memory_usage(deep=True).sum() / (1024 * 1024)
    del long_df  # Free the long DataFrame (can be large)
    gc.collect()  # Force garbage collection
    print(f"[INFO] Freed ~{long_df_size_mb:.1f}MB from long_df")
    
    # Reload long_df if needed for later steps (will be reloaded in write_outputs if needed)
    
    para_df = pd.DataFrame()
    print("[STEP] Chunking full-contract TXT files...")
    para_df = build_contract_paragraph_index(
        txt_dir=FULL_CONTRACTS_TXT_DIR,
        out_path=os.path.join(args.out_dir, "cuad_paragraph_index.parquet"),
        chunk_chars=args.chunk_chars,
        overlap=args.chunk_overlap,
    )

    # Optional: Flatten SQuAD JSON for extractive reader fine-tuning
    squad_df = pd.DataFrame()
    if CUAD_JSON_FILEPATH and Path(CUAD_JSON_FILEPATH).exists():
        print("[STEP] Loading SQuAD-style JSON...")
        squad_df = load_squad_json(CUAD_JSON_FILEPATH)

    # Write outputs (reload long_df from temp file)
    print("[STEP] Writing outputs...")
    long_df_temp_path = os.path.join(args.out_dir, "cuad_long_clauses_temp.parquet")
    if os.path.exists(long_df_temp_path):
        long_df = pd.read_parquet(long_df_temp_path)
        write_outputs(args.out_dir, long_df, squad_df, para_df)
        # Clean up temp file
        os.remove(long_df_temp_path)
    else:
        # Fallback: long_df should still be in memory
        write_outputs(args.out_dir, long_df, squad_df, para_df)

    # Quick report
    print("\n[REPORT]")
    print(f" Clauses rows: {len(long_df):,} (train={sum(long_df['split']=='train'):,}, val={sum(long_df['split']=='val'):,}, test={sum(long_df['split']=='test'):,})")
    print(f" Unique contracts: {long_df['filename'].nunique():,}")
    print(f" Categories covered: {long_df['category'].nunique()}")
    if not para_df.empty:
        print(f" Paragraph chunks: {len(para_df):,}")
    if not squad_df.empty:
        print(f" SQuAD QA rows: {len(squad_df):,}")

if __name__ == "__main__":
    main()
