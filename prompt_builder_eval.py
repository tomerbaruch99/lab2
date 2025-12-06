"""
Evaluation Prompt Builder for Haifa RAG
Creates deterministic, comparable prompts across chunking strategies and retrievers.
"""

from typing import List, Dict


class PromptBuilderEval:
    def __init__(self, max_chunk_length: int = 450):
        self.max_chunk_length = max_chunk_length

    def build(self, question: str, chunks: List[Dict]) -> str:
        parts = []
        parts.append("ענה אך ורק על בסיס המידע הבא.\n")

        parts.append("==== CONTEXT ====")
        if not chunks:
            parts.append("(no context retrieved)")
        else:
            for i, c in enumerate(chunks, 1):
                text = c.get("chunk_text_only") or c.get("text", "")
                text = text[: self.max_chunk_length]
                parts.append(f"[{i}] {text}")

        parts.append("\n==== QUESTION ====")
        parts.append(question)

        parts.append("\n==== REQUIRED OUTPUT FORMAT ====")
        parts.append("תשובה:")
        parts.append("מקורות (רק URLs):")
        parts.append("הסבר קצר:")
        parts.append("==== END ====")

        return "\n".join(parts)
