"""
Prompt Builder for Haifa Municipality RAG
=========================================
Builds prompts for LLM with retrieved chunks from Pinecone.

Fully updated to support:
- Hebrew data
- hyperlinks list from metadata
- doc_type
- chunking_strategy
- namespace
- robust URL extraction
"""

from typing import List, Dict, Optional
from enum import Enum
import json
import re


class PromptStyle(Enum):
    DETAILED = "detailed"        # Full context + metadata
    CONCISE = "concise"          # Shortened minimal context
    CONVERSATIONAL = "conv"      # Conversational format
    STRUCTURED = "structured"    # Highly structured (for advanced usage)
    EVAL = "eval"                # For evaluation purposes (file #3)


class PromptBuilder:
    """Build RAG prompts for several styles."""

    RE_URL_PATTERN = re.compile(r'\[URL:\s*([^\]]+)\]')

    def __init__(
        self,
        system_instruction: Optional[str] = None,
        style: PromptStyle = PromptStyle.DETAILED,
        max_chunk_length: int = 550,
        include_metadata: bool = True,
        include_hyperlinks: bool = True
    ):
        self.style = style
        self.max_chunk_length = max_chunk_length
        self.include_metadata = include_metadata
        self.include_hyperlinks = include_hyperlinks
        self.system_instruction = system_instruction or self._default_system_instruction()

    # -----------------------
    # DEFAULT SYSTEM PROMPT
    # -----------------------
    def _default_system_instruction(self) -> str:
        return (
            "אתה עוזר AI רשמי של עיריית חיפה. "
            "עליך לענות על שאלות רק מתוך מידע שנמצא במסמכים שסופקו לך. "
            "אל תמציא מידע. "
            "אם אינך יודע – אמור שאינך יודע. "
            "ענה בעברית בלבד. "
            "כאשר אתה מזכיר שירות, טופס או סעיף שיש לו קישור – כלול את הקישור."
        )

    # -----------------------
    # LINK EXTRACTION
    # -----------------------
    def _extract_hyperlinks_from_chunk(self, chunk: Dict) -> List[str]:
        links = []

        # 1. Try metadata field "hyperlinks"
        raw = chunk.get("hyperlinks")
        if raw:
            try:
                meta_links = raw if isinstance(raw, list) else json.loads(raw)
                for url in meta_links:
                    url = url.strip()
                    if url and url not in links:
                        links.append(url)
            except Exception:
                pass

        # 2. Extract from text "[URL: ...]"
        for field in ("chunk_text_only", "text"):
            text = chunk.get(field, "")
            if text:
                for url in self.RE_URL_PATTERN.findall(text):
                    url = url.strip()
                    if url and url not in links:
                        links.append(url)

        return links

    # -----------------------
    # PUBLIC MAIN BUILDER
    # -----------------------
    def build_prompt(
        self,
        question: str,
        chunks: List[Dict],
        include_sources: bool = True,
        custom_instruction: Optional[str] = None,
    ) -> str:

        if self.style == PromptStyle.DETAILED:
            return self._build_detailed(question, chunks, include_sources, custom_instruction)

        if self.style == PromptStyle.CONCISE:
            return self._build_concise(question, chunks, include_sources, custom_instruction)

        if self.style == PromptStyle.CONVERSATIONAL:
            return self._build_conversational(question, chunks, include_sources, custom_instruction)

        if self.style == PromptStyle.STRUCTURED:
            return self._build_structured(question, chunks, include_sources, custom_instruction)

        if self.style == PromptStyle.EVAL:
            return self._build_eval_prompt(question, chunks)

        return self._build_detailed(question, chunks, include_sources, custom_instruction)

    # -----------------------
    # DETAILED PROMPT
    # -----------------------
    def _build_detailed(self, question, chunks, include_sources, custom_instruction):
        parts = []
        parts.append(custom_instruction or self.system_instruction)
        parts.append("")
        parts.append("=" * 60)
        parts.append("מידע רלוונטי מאתר עיריית חיפה:")
        parts.append("=" * 60)
        parts.append("")

        if not chunks:
            parts.append("לא נמצא מידע רלוונטי.")
        else:
            for i, chunk in enumerate(chunks, 1):
                parts.append(f"[מקור {i}]")

                if self.include_metadata:
                    if chunk.get("title"):
                        parts.append(f"כותרת: {chunk['title']}")
                    if chunk.get("subtitle"):
                        parts.append(f"תת-כותרת: {chunk['subtitle']}")
                    if chunk.get("url"):
                        parts.append(f"URL: {chunk['url']}")
                    if chunk.get("chunking_strategy"):
                        parts.append(f"Chunking: {chunk['chunking_strategy']}")
                    if chunk.get("doc_type"):
                        parts.append(f"סוג קובץ: {chunk['doc_type']}")
                    if chunk.get("namespace"):
                        parts.append(f"Namespace: {chunk['namespace']}")

                parts.append("")

                body = chunk.get("chunk_text_only") or chunk.get("text", "")
                if len(body) > self.max_chunk_length:
                    body = body[: self.max_chunk_length] + "..."
                parts.append(body)

                # hyperlinks
                if self.include_hyperlinks:
                    links = self._extract_hyperlinks_from_chunk(chunk)
                    if links:
                        parts.append("\nקישורים חשובים:")
                        for url in links[:8]:
                            parts.append(f"- {url}")

                if i < len(chunks):
                    parts.append("\n" + "-" * 50 + "\n")

        parts.append("\n" + "=" * 60)
        parts.append("שאלת המשתמש:")
        parts.append(question)
        parts.append("ענה על בסיס המידע בלבד.")

        return "\n".join(parts)

    # -----------------------
    # CONCISE PROMPT
    # -----------------------
    def _build_concise(self, question, chunks, include_sources, custom_instruction):
        parts = []
        parts.append(custom_instruction or "ענה על השאלה לפי המידע הבא:")

        for i, c in enumerate(chunks, 1):
            t = c.get("chunk_text_only") or c.get("text", "")
            if len(t) > self.max_chunk_length:
                t = t[: self.max_chunk_length] + "..."
            line = f"{i}. {t}"
            if include_sources and c.get("url"):
                line += f" ({c['url']})"
            parts.append(line)

        parts.append("")
        parts.append(f"שאלה: {question}")
        return "\n".join(parts)

    # -----------------------
    # CONVERSATIONAL
    # -----------------------
    def _build_conversational(self, question, chunks, include_sources, custom_instruction):
        parts = []
        parts.append("היי! מצאתי את המידע הבא מתוך אתר עיריית חיפה:")

        for c in chunks:
            t = c.get("chunk_text_only") or c.get("text", "")
            t = t[: self.max_chunk_length] + ("..." if len(t) > self.max_chunk_length else "")
            line = "- " + t
            if include_sources and c.get("url"):
                line += f" ({c['url']})"
            parts.append(line)

        parts.append("")
        parts.append(f"שאלתך: {question}")
        parts.append("הנה התשובה על בסיס המקורות.")
        return "\n".join(parts)

    # -----------------------
    # STRUCTURED
    # -----------------------
    def _build_structured(self, question, chunks, include_sources, custom_instruction):
        parts = []
        parts.append("=== SYSTEM ===")
        parts.append(custom_instruction or self.system_instruction)
        parts.append("\n=== CONTEXT ===")

        for i, c in enumerate(chunks, 1):
            parts.append(f"[Source {i}]")
            if include_sources and c.get("url"):
                parts.append(f"URL: {c['url']}")
            t = c.get("chunk_text_only") or c.get("text", "")
            t = t[: self.max_chunk_length] + ("..." if len(t) > self.max_chunk_length else "")
            parts.append(t)
            parts.append("")

        parts.append("=== QUESTION ===")
        parts.append(question)
        parts.append("=== ANSWER ===")
        return "\n".join(parts)

    # -----------------------
    # (3) EVALUATION PROMPT — *CONSISTENT, MEASURABLE, NEUTRAL*
    # -----------------------
    def _build_eval_prompt(self, question, chunks):
        """
        Designed specifically for academic evaluation.
        Produces deterministic, short, machine-checkable structure.
        """

        parts = []
        parts.append("ענה אך ורק על בסיס המידע הבא. אל תוסיף שום ידע חיצוני.\n")

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

        parts.append("\n==== FORMAT REQUIREMENTS ====")
        parts.append("תשובה:")
        parts.append("מקורות (URLs בלבד):")
        parts.append("הסבר קצר:")
        parts.append("\n==== END ====")

        return "\n".join(parts)
