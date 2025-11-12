"""
Extractive Reader for Contract Clause Q&A

Fine-tunes an extractive QA model on CUAD SQuAD-style JSON to return exact answer spans
with start/end indices. Useful for extracting dates, Yes/No answers, governing law, etc.
"""

import os
import re
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import Dataset

# Configuration
BASE_MODEL = "microsoft/deberta-v3-base"  # Good for extractive QA
OUTPUT_DIR = "models/extractive_reader"
SQUAD_PARQUET_PATH = "cuad_prepared_data/cuad_squad_flat.parquet"

# Redaction pattern
RE_REDACT = re.compile(r"(\*{2,}|_{2,}|<omitted>)", re.IGNORECASE)


def load_cuad_squad(parquet_path: str) -> Dataset:
    """
    Load CUAD SQuAD-style data from parquet file.
    
    Expected columns: question, context, answer_text, answer_start, is_impossible
    Returns: HuggingFace Dataset with 'question', 'context', 'answers' columns
    """
    df = pd.read_parquet(parquet_path)
    
    # Filter out empty contexts
    df = df[df["context"].str.strip() != ""].copy()
    
    # Convert to SQuAD format: answers should be a dict with 'text' and 'answer_start' lists
    def format_answers(row):
        if row["answer_start"] >= 0 and pd.notna(row["answer_text"]):
            return {
                "text": [str(row["answer_text"])],
                "answer_start": [int(row["answer_start"])]
            }
        else:
            return {"text": [], "answer_start": []}
    
    df["answers"] = df.apply(format_answers, axis=1)
    
    # Select required columns
    dataset = Dataset.from_pandas(df[["question", "context", "answers"]])
    return dataset


def preprocess_function(examples, tokenizer, max_length=384, stride=128):
    """
    Tokenize examples for extractive QA.
    Handles truncation and offset mapping for answer span extraction.
    """
    # Tokenize with truncation on context only
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    
    # Map answer spans to token positions
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]
        
        cls_index = tokenized["input_ids"][i].index(tokenizer.cls_token_id)
        
        # Find start and end token positions
        start_char = answers["answer_start"][0] if answers["answer_start"] else None
        end_char = start_char + len(answers["text"][0]) if start_char is not None and answers["text"] else None
        
        if start_char is not None and end_char is not None:
            # Find token positions
            token_start_index = 0
            token_end_index = len(offsets) - 1
            
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            token_start_index -= 1
            
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            token_end_index += 1
            
            # If answer is out of span, set to CLS token
            if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
                token_start_index = cls_index
                token_end_index = cls_index
        else:
            # No answer (impossible question)
            token_start_index = cls_index
            token_end_index = cls_index
        
        tokenized["start_positions"].append(token_start_index)
        tokenized["end_positions"].append(token_end_index)
    
    return tokenized


def train_extractive_reader(
    squad_parquet_path: str = SQUAD_PARQUET_PATH,
    base_model: str = BASE_MODEL,
    output_dir: str = OUTPUT_DIR,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 384,
    stride: int = 128,
    test_size: float = 0.15,
    seed: int = 42
):
    """
    Fine-tune an extractive QA model on CUAD SQuAD data.
    
    Args:
        squad_parquet_path: Path to CUAD SQuAD flat parquet file
        base_model: Base model to fine-tune
        output_dir: Directory to save fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
        stride: Stride for sliding window
        test_size: Fraction of data for validation
        seed: Random seed
    """
    print(f"[INFO] Loading CUAD SQuAD data from {squad_parquet_path}...")
    dataset = load_cuad_squad(squad_parquet_path)
    
    print(f"[INFO] Dataset size: {len(dataset)}")
    
    # Split into train/test
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    
    # Load tokenizer
    print(f"[INFO] Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    
    # Preprocess datasets
    print("[INFO] Preprocessing datasets...")
    train_dataset = dataset["train"].map(
        lambda x: preprocess_function(x, tokenizer, max_length, stride),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    eval_dataset = dataset["test"].map(
        lambda x: preprocess_function(x, tokenizer, max_length, stride),
        batched=True,
        remove_columns=dataset["test"].column_names
    )
    
    # Load model
    print(f"[INFO] Loading model from {base_model}...")
    model = AutoModelForQuestionAnswering.from_pretrained(base_model)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        save_total_limit=2,
        seed=seed
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )
    
    # Train
    print("[INFO] Starting training...")
    trainer.train()
    
    # Save model
    print(f"[INFO] Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"[✓] Training complete! Model saved to {output_dir}")


class ExtractiveReader:
    """
    Extractive QA reader for contract clauses.
    Returns exact answer spans with start/end indices.
    """
    
    def __init__(self, model_dir: str = OUTPUT_DIR, device: str = None):
        """
        Initialize extractive reader.
        
        Args:
            model_dir: Path to fine-tuned model directory
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def extract_answer(
        self,
        question: str,
        context: str,
        max_length: int = 384,
        stride: int = 128,
        return_score: bool = True
    ) -> dict:
        """
        Extract answer span from context given a question.
        
        Args:
            question: Question string
            context: Context string (clause text)
            max_length: Maximum sequence length
            stride: Stride for sliding window
            return_score: Whether to return confidence score
        
        Returns:
            Dictionary with:
                - answer_text: Extracted answer span
                - start: Start character index in context
                - end: End character index in context
                - score: Confidence score (if return_score=True)
                - has_redaction: Whether answer contains redaction markers
        """
        # Tokenize
        encodings = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=max_length,
            stride=stride,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True
        )
        
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        # Get predictions
        outputs = self.model(**encodings)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        
        # Get best span
        start_idx = int(torch.argmax(start_logits))
        end_idx = int(torch.argmax(end_logits))
        
        # Calculate score
        score = (start_logits[start_idx].item() + end_logits[end_idx].item()) / 2.0
        
        # Map back to character positions
        offset_mapping = encodings["offset_mapping"][0].cpu().tolist()
        start_char = offset_mapping[start_idx][0]
        end_char = offset_mapping[end_idx][1]
        
        # Extract answer text
        if 0 <= start_char < end_char <= len(context):
            answer_text = context[start_char:end_char].strip()
        else:
            answer_text = ""
        
        # Check for redaction
        has_redaction = bool(RE_REDACT.search(answer_text)) if answer_text else False
        
        result = {
            "answer_text": answer_text,
            "start": start_char,
            "end": end_char,
            "has_redaction": has_redaction
        }
        
        if return_score:
            result["score"] = score
        
        return result
    
    @torch.no_grad()
    def answer_over_clauses(
        self,
        question: str,
        clauses: list,
        max_length: int = 384,
        stride: int = 128
    ) -> dict:
        """
        Extract answer from multiple clauses, returning the best match.
        
        Args:
            question: Question string
            clauses: List of clause dictionaries with keys:
                - text: Clause text (required)
                - filename: Filename (optional)
                - category: Category (optional)
                - clause_id: Clause ID (optional)
            max_length: Maximum sequence length
            stride: Stride for sliding window
        
        Returns:
            Best answer dictionary with metadata, or None if no valid answer found
        """
        best_answer = None
        best_score = float('-inf')
        
        for clause in clauses:
            clause_text = clause.get("text", clause.get("context", ""))
            if not clause_text:
                continue
            
            result = self.extract_answer(question, clause_text, max_length, stride)
            
            # Skip empty answers
            if not result["answer_text"]:
                continue
            
            score = result.get("score", 0.0)
            
            if score > best_score:
                best_score = score
                best_answer = {
                    "answer_text": result["answer_text"],
                    "start": result["start"],
                    "end": result["end"],
                    "score": score,
                    "filename": clause.get("filename"),
                    "category": clause.get("category"),
                    "clause_id": clause.get("clause_id"),
                    "has_redaction": result["has_redaction"]
                }
        
        return best_answer if best_answer and best_answer["score"] >= 0 else None

