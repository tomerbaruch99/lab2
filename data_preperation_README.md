# CUAD RAG Preparation (Stages 0 & 1)

This package sets up the data you need for two products built on CUAD:
1. **Contract Clause Q&A Assistant** (retrieve clauses → extract/answer → cite evidence)
2. **RAG-driven Contract Reviewer** (presence checks, outlier detection baselines)

## What you get
- `cuad_long_clauses.parquet` & `cuad_long_clauses.jsonl` — flattened clause-level corpus for retrieval (one row per category/contract clause).
- `cuad_paragraph_index.parquet` (optional) — chunked paragraph windows from the full TXT contracts for paragraph-level fallback retrieval.
- `cuad_squad_flat.parquet` (optional) — SQuAD-style QA rows for extractive reader fine-tuning.

## Usage
```bash
python cUAD_prep.py \
  --csv /path/to/CUAD_v1.csv \
  --squad_json /path/to/CUAD_v1.json \  # optional
  --txt_dir /path/to/full_contracts_txt \  # optional
  --out_dir ./cuad_prepared \
  --chunk_chars 4000 \
  --chunk_overlap 400 \
  --val_size 0.15 \
  --test_size 0.15
```

## Outputs
- **RAG-ready clauses**: JSONL lines with `text` and `metadata` (filename, contract_type, category, question_template, answer, answer_type)
- **Splits**: train/val/test assigned at the contract level and joined into the clause rows.

## Notes
- The script is **redaction-aware**: it preserves `***`, `___`, and `<omitted>` and never attempts to impute masked content.
- CSV must have 83 columns with `-Answer` pairs (41 categories + filename).
- You can extend `QUESTION_TEMPLATES` for richer user prompts.
