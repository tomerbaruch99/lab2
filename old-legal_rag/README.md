# 0) Run your existing ETL (you already have it)
python data_preparation.py --out_dir ./cuad_prepared_data

# 1) Build index (FAISS local)
python -m src.indexing_cuad --split all --use_para_index --save_faiss ./cuad_faiss.index

#    or Pinecone
export PINECONE_API_KEY=...
export VSTORE=pinecone
python -m src.indexing_cuad --split all --use_para_index

# 2) Quick query
python -m src.rag_service --q "Is there an exclusivity obligation?" --k 8

# 3) Create test set from CUAD's own questions/answers
python -m src.create_testset_cuad --use_split test --n_per_category 25 --out_csv ./testsets/casebank.csv

# 4) Evaluate (objective + simple correctness)
python -m src.evaluate_cuad --test_csv ./testsets/casebank.csv --out_csv ./evaluation/eval_results.csv


## category-aware search
# Baseline
python -m src.evaluate_cuad \
  --test_csv ./testsets/casebank.csv \
  --out_csv ./evaluation/eval_off.csv \
  --category_router_mode off

# Hard
python -m src.evaluate_cuad \
  --test_csv ./testsets/casebank.csv \
  --out_csv ./evaluation/eval_hard.csv \
  --category_router_mode hard \
  --category_hard_bonus 0.20

# Soft
python -m src.evaluate_cuad \
  --test_csv ./testsets/casebank.csv \
  --out_csv ./evaluation/eval_soft.csv \
  --category_router_mode soft \
  --category_soft_max_bonus 0.12