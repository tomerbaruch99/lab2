# This file gets cuad_squad_flat.parquet from data_preparation.py and trains a QA model

import os, torch, pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator

MODEL_NAME = "deberta-v3-base"
OUT_DIR = "models/extractive_reader"

def load_cuad_squad(parquet_path):
    df = pd.read_parquet(parquet_path)  # cols: question, context, answer_text, answer_start, is_impossible
    df = df[df["context"].str.strip()!=""].copy()
    df["answers"] = df.apply(lambda r: {"text":[r["answer_text"]], "answer_start":[int(r["answer_start"])]} 
                             if r["answer_start"]>=0 else {"text":[], "answer_start":[]}, axis=1)
    return Dataset.from_pandas(df[["question","context","answers"]])

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
def preprocess(examples, max_len=384, doc_stride=128):
    return tok(examples["question"], examples["context"], truncation="only_second",
               max_length=max_len, stride=doc_stride, return_overflowing_tokens=True,
               return_offsets_mapping=True, padding="max_length")

def main():
    ds = load_cuad_squad("cuad_prepared/cuad_squad_flat.parquet")
    ds = ds.train_test_split(test_size=0.15, seed=42)
    tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    args = TrainingArguments(
        output_dir=OUT_DIR, learning_rate=2e-5, per_device_train_batch_size=8,
        per_device_eval_batch_size=8, num_train_epochs=3, weight_decay=0.01,
        evaluation_strategy="epoch", save_strategy="epoch", fp16=torch.cuda.is_available(),
        logging_steps=50, load_best_model_at_end=True, metric_for_best_model="loss"
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"],
                      eval_dataset=tokenized["test"], tokenizer=tok,
                      data_collator=default_data_collator)
    trainer.train()
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

if __name__ == "__main__":
    main()
