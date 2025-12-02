# Overview

It’s a command-line Python ETL program that:

1. reads a CUAD “master” CSV file;
2. converts it into a **long** table of clause records (one row per (file, category));
3. heuristically infers a contract type per file;
4. cleans text, lightly normalizes answers, and attaches a ready-made question template per category;
5. creates **train/val/test** splits at the contract (filename) level;
6. optionally **chunks** raw TXT contracts into overlapping character windows;
7. optionally **flattens** a SQuAD-style JSON into rows;
8. writes multiple output artifacts (Parquet, JSONL, etc.) and prints a small report.

All file locations come from `consts`:

* `CUAD_CSV_FILEPATH`
* `CUAD_JSON_FILEPATH`
* `FULL_CONTRACTS_TXT_DIR`

# Important constants

* `CUAD_CATEGORIES`: a list of dicts with keys `name`, `answer_type`, `group`. Used to look up the `answer_type` for each category.
* `QUESTION_TEMPLATES`: mapping from category name → a natural-language question. If a category isn’t in this dict, a fallback “What is the answer for category …?” is used.
* `CONTRACT_TYPES`: a list of contract type phrases; used to infer `contract_type`.

# Regex/cleaning utilities

Compiled patterns:

* `RE_WHITESPACE`: collapses runs of spaces/tabs.
* `RE_PAGE_FOOTER`: matches “page 1”, “page 1 of 10”, etc.
* `RE_HEADER_FOOTER_NOISE`: matches words like “confidential”, “exhibit”, “table of contents”, etc.
* `RE_RED_ACTION`: matches redaction tokens like `***`, `___`, or `<omitted>`.

`clean_clause_text(text)`:

* Returns `""` for `None`/NaN or string “nan”/“None”/empty.
* Otherwise: coerces to string, replaces `\r` with `\n`, collapses whitespace, removes header/footer noise and page markers. (It **does not** remove the redaction tokens matched by `RE_RED_ACTION`.)

`normalize_answer(category, answer)`:

* Calls the cleaner above.
* If empty → `""`.
* If `RE_RED_ACTION` is present → returns the answer unchanged.
* If the category’s `answer_type` (looked up in `CUAD_CATEGORIES`) is `yesno` and the answer is exactly “yes”/“no” (case-insensitive), it returns **capitalized** `Yes` or `No`. Otherwise it returns the stripped answer.

# Category & contract inference

`infer_categories_from_columns(columns)`:

* Walks the CSV header names in order (skipping the literal column `Filename`).
* For each “context” column `C`, it looks for a matching answer column named `C-Answer` (case-sensitive first, then a case-insensitive fallback). If found, yields `(C, answer_col)`; otherwise `(C, None)`.

`infer_contract_type(filename, document_name_answer)`:

* Builds a lowercase string from the provided `document_name_answer` and `filename` stem.
* Returns the first `contract_type` from `CONTRACT_TYPES` that is a substring of that string; otherwise `"unknown"`.

# Chunking

`chunk_text(text, max_chars=4000, overlap=400)`:

* Cleans the text, then emits overlapping character windows.
* Tries to end a chunk at the last “`. `” before the hard limit, unless that cut would be too early (≤ start+200), in which case it hard-cuts.
* Returns a list of `(start_char_index, chunk_text)`.

# CSV → long table

`read_master_csv(path)`:

* `pd.read_csv`, then strips whitespace off column names.

`build_long_table(df)`:

* Calls `infer_categories_from_columns` on the CSV header.
* For each row (contract) and for each `(context_col, answer_col)` pair:

  * `filename = row["Filename"]` (default `""` if missing).
  * `contract_type = infer_contract_type(filename, row["Document Name-Answer"] or "")`.
  * `context = clean_clause_text(row[context_col])`.
  * `answer_raw = row[answer_col]` (or `""` if no answer column); `answer = normalize_answer(category, answer_raw)`.
  * `answer_type` is looked up in `CUAD_CATEGORIES` by category name; default `"text"`.
  * `question_template` from `QUESTION_TEMPLATES`, with a fallback.
* Builds a DataFrame with columns:

  * `filename, contract_type, category, context, answer, answer_type, question_template`.
* Drops rows where `context` is empty/whitespace.

# Optional: TXT paragraph index

`build_contract_paragraph_index(txt_dir, out_path, chunk_chars, overlap)`:

* If the directory doesn’t exist, logs a warning and returns an empty DataFrame.
* Reads each `*.txt` file (UTF-8 with fallback to Latin-1), calls `chunk_text`, and accumulates rows with:

  * `doc_id` (filename stem), `filename`, `chunk_id` (sequential), `start_char`, `text`.
* Saves to `out_path`:

  * If suffix = `.parquet` → Parquet, else CSV.
* Returns the DataFrame (may be empty).

# Splitting

`make_splits(contract_df, val_size=0.15, test_size=0.15, seed=42)`:

* Collapses to unique `(filename, contract_type)` pairs.
* For **each** `contract_type` group independently:

  * Randomly shuffles indices using the provided seed.
  * Computes counts `n_test` and `n_val` by rounding `n * ratios`.
  * Assigns those many to “test” and “val”; the rest are “train”.
* Returns a mapping DataFrame with `filename, contract_type, split`.

`attach_splits(long_df, split_map)`:

* Left-merges on `filename, contract_type`.
* Missing splits default to `"train"`.

# Optional: SQuAD JSON flattening

`load_squad_json(path)`:

* Loads JSON and emits one row per **answer** (or one row with empty answer if none), with:

  * `filename` (JSON “title”),
  * `paragraph_id` = 32-bit hash of the paragraph “context”,
  * `question_id`, `question`, `answer_text`, `answer_start`, `is_impossible`, `context`.

# Writing outputs

`write_outputs(out_dir, long_df, squad_df, para_df)`:

* Ensures output dir exists.
* Writes:

  * `cuad_long_clauses.parquet` (the long table).
  * `cuad_long_clauses.jsonl` where each line has:

    * `id = "{filename}|{category}"`,
    * `text = context`,
    * `metadata = {filename, contract_type, category, answer_type, question_template, answer}`.
* If `squad_df` not empty → writes `cuad_squad_flat.parquet`.
* If `para_df` not empty → writes `cuad_paragraph_index.parquet`.

# Program entry point (`main`)

* Parses CLI args:

  * `--out_dir` (default `./cuad_prepared`)
  * `--chunk_chars` (default 4000)
  * `--chunk_overlap` (default 400)
  * `--val_size` (default 0.15)
  * `--test_size` (default 0.15)
* Loads the master CSV from `CUAD_CSV_FILEPATH`; verifies a `Filename` column exists.
* Builds the long table.
* Creates the split map and merges splits into the long table.
* Tries to build the TXT paragraph index from `FULL_CONTRACTS_TXT_DIR`, saving a CSV copy into `out_dir` and keeping the DataFrame in memory.
* If `CUAD_JSON_FILEPATH` is set and exists, loads it into a DataFrame.
* Calls `write_outputs(...)`.
* Prints a short report with counts (rows per split, unique contracts, number of categories, paragraph chunk count, SQuAD row count).

# Notable behaviors & edge handling

* **Cleaning** removes common header/footer noise and page markers, but preserves explicit redaction tokens (`***`, `___`, `<omitted>`).
* **Yes/No** answers are normalized only when the category’s `answer_type` equals `"yesno"` and the raw answer is exactly “yes” or “no”.
* **Category matching** is tolerant to case differences for the paired “-Answer” column.
* **Contract type** is a substring match against filename/doc-name; otherwise `"unknown"`.
* **Stratified splitting** is done per `contract_type`; small groups may lead to coarse splits because counts are rounded.
* Missing optional inputs (TXT/JSON) simply produce empty DataFrames and skip related writes.
* If any rows in the long table lack a matched split (e.g., due to merge issues), they’re assigned `"train"`.
