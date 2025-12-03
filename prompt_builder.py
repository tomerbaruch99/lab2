"""
Prompt Builder for Haifa Municipality RAG
=========================================
Builds prompts for LLM with retrieved chunks from Pinecone.

This module formats user questions and retrieved chunks into prompts
suitable for language models (Gemini, GPT, etc.).

Usage:
    from prompt_builder import PromptBuilder
    
    builder = PromptBuilder()
    prompt = builder.build_prompt(
        question="איך משלמים ארנונה?",
        chunks=retrieved_chunks,
        include_sources=True
    )
"""

from typing import List, Dict, Optional
from enum import Enum
import json
import re


class PromptStyle(Enum):
    """Different prompt styles for various use cases."""
    DETAILED = "detailed"  # Full context with sources
    CONCISE = "concise"   # Minimal formatting
    CONVERSATIONAL = "conversational"  # Natural conversation style
    STRUCTURED = "structured"  # Highly structured format


class PromptBuilder:
    """
    Builds prompts for RAG-based question answering.
    
    Formats user questions with retrieved chunks into prompts
    suitable for language models.
    """
    
    # Regex to extract URLs from [URL: ...] format
    RE_URL_PATTERN = re.compile(r'\[URL:\s*([^\]]+)\]')
    
    def __init__(
        self,
        system_instruction: Optional[str] = None,
        style: PromptStyle = PromptStyle.DETAILED,
        max_chunk_length: int = 500,
        include_metadata: bool = True,
    ):
        """
        Initialize the prompt builder.
        
        Args:
            system_instruction: Custom system instruction (default: municipality-specific)
            style: Prompt style to use
            max_chunk_length: Maximum characters to show per chunk in prompt
            include_metadata: Whether to include URLs, titles, etc.
        """
        self.style = style
        self.max_chunk_length = max_chunk_length
        self.include_metadata = include_metadata
        
        if system_instruction:
            self.system_instruction = system_instruction
        else:
            self.system_instruction = self._default_system_instruction()
    
    def _extract_hyperlinks_from_chunk(self, chunk: Dict) -> List[str]:
        """
        Extract hyperlinks from a chunk.
        
        Args:
            chunk: Chunk dictionary with text content
        
        Returns:
            List of unique URLs found in the chunk
        """
        hyperlinks = []
        
        # Try to get hyperlinks from metadata (if stored during data preparation)
        if chunk.get("hyperlinks"):
            try:
                if isinstance(chunk["hyperlinks"], str):
                    hyperlinks = json.loads(chunk["hyperlinks"])
                elif isinstance(chunk["hyperlinks"], list):
                    hyperlinks = chunk["hyperlinks"]
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Also extract from text content (fallback)
        text_content = chunk.get("chunk_text_only") or chunk.get("text", "")
        if text_content:
            urls_from_text = self.RE_URL_PATTERN.findall(str(text_content))
            for url in urls_from_text:
                url = url.strip()
                if url and url not in hyperlinks:
                    hyperlinks.append(url)
        
        return hyperlinks
    
    def _default_system_instruction(self) -> str:
        """Default system instruction for Haifa municipality chatbot."""
        return """אתה עוזר AI מומחה של עיריית חיפה. תפקידך לענות על שאלות של תושבים על בסיס המידע הרשמי מאתר העירייה.

הוראות:
1. ענה בעברית בצורה ברורה, מקצועית וידידותית
2. השתמש רק במידע מהמסמכים שסופקו לך
3. אם אינך יודע את התשובה, אמור זאת בכנות
4. ציין את המקורות (URLs) כשאתה מצטט מידע
5. אם יש מספר מקורות, ציין את כולם
6. שמור על דיוק - אל תמציא מידע שלא קיים במסמכים
7. חשוב: אסור לתת ייעוץ משפטי או פיננסי מעבר לתיאור מה שכתוב במסמכים. תאר את המידע הקיים במסמכים, אך אל תמליץ או תייעץ מעבר לכך
8. כאשר אתה מזכיר שירות, טופס, או מידע שיש לו קישור (hyperlink) במסמכים, הקפד לכלול את הקישור הרלוונטי בתשובתך. הקישורים מסומנים כ-[URL: ...] בטקסט או מופיעים ברשימת הקישורים של כל מקור"""
    
    def build_prompt(
        self,
        question: str,
        chunks: List[Dict],
        include_sources: bool = True,
        custom_instruction: Optional[str] = None,
    ) -> str:
        """
        Build a prompt from question and retrieved chunks.
        
        Args:
            question: User's question
            chunks: List of retrieved chunks (from retriever)
            include_sources: Whether to include source URLs/titles
            custom_instruction: Optional custom instruction to prepend
        
        Returns:
            Formatted prompt string
        """
        if self.style == PromptStyle.DETAILED:
            return self._build_detailed_prompt(question, chunks, include_sources, custom_instruction)
        elif self.style == PromptStyle.CONCISE:
            return self._build_concise_prompt(question, chunks, include_sources, custom_instruction)
        elif self.style == PromptStyle.CONVERSATIONAL:
            return self._build_conversational_prompt(question, chunks, include_sources, custom_instruction)
        elif self.style == PromptStyle.STRUCTURED:
            return self._build_structured_prompt(question, chunks, include_sources, custom_instruction)
        else:
            return self._build_detailed_prompt(question, chunks, include_sources, custom_instruction)
    
    def _build_detailed_prompt(
        self,
        question: str,
        chunks: List[Dict],
        include_sources: bool,
        custom_instruction: Optional[str],
    ) -> str:
        """Build a detailed prompt with full context."""
        parts = []
        
        # System instruction
        instruction = custom_instruction or self.system_instruction
        parts.append(instruction)
        parts.append("")
        
        # Context section
        parts.append("=" * 60)
        parts.append("מידע רלוונטי מאתר עיריית חיפה:")
        parts.append("=" * 60)
        parts.append("")
        
        if not chunks:
            parts.append("לא נמצא מידע רלוונטי במאגר המידע.")
        else:
            for i, chunk in enumerate(chunks, 1):
                chunk_parts = []
                
                # Chunk header
                if include_sources and self.include_metadata:
                    chunk_parts.append(f"[מקור {i}]")
                    if chunk.get("title"):
                        chunk_parts.append(f"כותרת: {chunk['title']}")
                    if chunk.get("subtitle"):
                        chunk_parts.append(f"תת-כותרת: {chunk['subtitle']}")
                    if chunk.get("url"):
                        chunk_parts.append(f"קישור: {chunk['url']}")
                    if chunk.get("score") is not None:
                        chunk_parts.append(f"רלוונטיות: {chunk['score']:.3f}")
                    chunk_parts.append("")
                
                # Chunk content
                content = chunk.get("chunk_text_only") or chunk.get("text", "")
                if len(content) > self.max_chunk_length:
                    content = content[:self.max_chunk_length] + "..."
                chunk_parts.append(content)
                
                # Extract and include hyperlinks from the chunk
                hyperlinks = self._extract_hyperlinks_from_chunk(chunk)
                if hyperlinks:
                    chunk_parts.append("")
                    chunk_parts.append("קישורים רלוונטיים:")
                    for link in hyperlinks[:10]:  # Limit to 10 links per chunk
                        chunk_parts.append(f"  - {link}")
                
                parts.append("\n".join(chunk_parts))
                if i < len(chunks):
                    parts.append("")
                    parts.append("-" * 60)
                    parts.append("")
        
        parts.append("")
        parts.append("=" * 60)
        parts.append("")
        
        # Question section
        parts.append("שאלת המשתמש:")
        parts.append(question)
        parts.append("")
        parts.append("בבקשה ענה על השאלה על בסיס המידע שסופק לעיל.")
        
        if include_sources:
            parts.append("אם אתה מצטט מידע, ציין את המקור (URL) של המסמך.")
            parts.append("כאשר אתה מזכיר שירות, טופס, או פעולה שיש לה קישור במסמכים, הקפד לכלול את הקישור הרלוונטי בתשובתך.")
        
        return "\n".join(parts)
    
    def _build_concise_prompt(
        self,
        question: str,
        chunks: List[Dict],
        include_sources: bool,
        custom_instruction: Optional[str],
    ) -> str:
        """Build a concise prompt with minimal formatting."""
        parts = []
        
        if custom_instruction:
            parts.append(custom_instruction)
        else:
            parts.append("ענה על השאלה הבאה על בסיס המידע שסופק:")
        
        parts.append("")
        parts.append("מידע:")
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("chunk_text_only") or chunk.get("text", "")
            if len(content) > self.max_chunk_length:
                content = content[:self.max_chunk_length] + "..."
            
            chunk_line = f"{i}. "
            if include_sources and chunk.get("url"):
                chunk_line += f"[{chunk['url']}] "
            chunk_line += content
            
            # Add hyperlinks if available
            hyperlinks = self._extract_hyperlinks_from_chunk(chunk)
            if hyperlinks:
                chunk_line += f" [קישורים: {', '.join(hyperlinks[:3])}]"
            
            parts.append(chunk_line)
        
        parts.append("")
        parts.append(f"שאלה: {question}")
        
        return "\n".join(parts)
    
    def _build_conversational_prompt(
        self,
        question: str,
        chunks: List[Dict],
        include_sources: bool,
        custom_instruction: Optional[str],
    ) -> str:
        """Build a conversational, natural prompt."""
        parts = []
        
        parts.append("שלום! אני כאן כדי לעזור לך עם שאלות על עיריית חיפה.")
        parts.append("")
        parts.append("הנה המידע הרלוונטי שמצאתי:")
        parts.append("")
        
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("chunk_text_only") or chunk.get("text", "")
            if len(content) > self.max_chunk_length:
                content = content[:self.max_chunk_length] + "..."
            
            chunk_info = []
            if chunk.get("title"):
                chunk_info.append(f"מתוך: {chunk['title']}")
            if include_sources and chunk.get("url"):
                chunk_info.append(f"({chunk['url']})")
            
            if chunk_info:
                parts.append(f"• {' | '.join(chunk_info)}")
            parts.append(f"  {content}")
            
            # Add hyperlinks if available
            hyperlinks = self._extract_hyperlinks_from_chunk(chunk)
            if hyperlinks:
                parts.append(f"  קישורים: {', '.join(hyperlinks[:5])}")
            
            parts.append("")
        
        parts.append(f"תבסס על המידע הזה, {question}")
        
        return "\n".join(parts)
    
    def _build_structured_prompt(
        self,
        question: str,
        chunks: List[Dict],
        include_sources: bool,
        custom_instruction: Optional[str],
    ) -> str:
        """Build a highly structured prompt."""
        parts = []
        
        # Header
        parts.append("=== מערכת RAG - עיריית חיפה ===")
        parts.append("")
        
        # Instruction
        instruction = custom_instruction or self.system_instruction
        parts.append("הוראות מערכת:")
        parts.append(instruction)
        parts.append("")
        
        # Context
        parts.append("=== הקשר (Context) ===")
        if not chunks:
            parts.append("אין מידע זמין.")
        else:
            parts.append(f"מספר מקורות: {len(chunks)}")
            parts.append("")
            
            for i, chunk in enumerate(chunks, 1):
                parts.append(f"--- מקור {i} ---")
                
                metadata = []
                if chunk.get("title"):
                    metadata.append(f"כותרת: {chunk['title']}")
                if chunk.get("subtitle"):
                    metadata.append(f"תת-כותרת: {chunk['subtitle']}")
                if include_sources and chunk.get("url"):
                    metadata.append(f"URL: {chunk['url']}")
                if chunk.get("score") is not None:
                    metadata.append(f"Score: {chunk['score']:.4f}")
                
                if metadata:
                    parts.append(" | ".join(metadata))
                
                content = chunk.get("chunk_text_only") or chunk.get("text", "")
                if len(content) > self.max_chunk_length:
                    content = content[:self.max_chunk_length] + "..."
                parts.append(content)
                
                # Add hyperlinks if available
                hyperlinks = self._extract_hyperlinks_from_chunk(chunk)
                if hyperlinks:
                    parts.append(f"קישורים: {', '.join(hyperlinks[:5])}")
                
                parts.append("")
        
        # Question
        parts.append("=== שאלה ===")
        parts.append(question)
        parts.append("")
        
        # Output format
        parts.append("=== פורמט תשובה ===")
        parts.append("1. תשובה ישירה לשאלה")
        if include_sources:
            parts.append("2. מקורות (URLs)")
        parts.append("3. פרטים נוספים (אם רלוונטי)")
        
        return "\n".join(parts)
    
    def build_chat_prompt(
        self,
        conversation_history: List[Dict[str, str]],
        chunks: List[Dict],
        include_sources: bool = True,
    ) -> str:
        """
        Build a prompt for conversational chat with history.
        
        Args:
            conversation_history: List of {"role": "user"/"assistant", "content": "..."}
            chunks: Retrieved chunks for current turn
            include_sources: Whether to include sources
        
        Returns:
            Formatted prompt with conversation history
        """
        parts = []
        
        # System instruction
        parts.append(self.system_instruction)
        parts.append("")
        parts.append("שיחה קודמת:")
        parts.append("-" * 60)
        
        # Add conversation history
        for msg in conversation_history[-5:]:  # Last 5 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"משתמש: {content}")
            elif role == "assistant":
                parts.append(f"עוזר: {content}")
        
        parts.append("-" * 60)
        parts.append("")
        
        # Current context
        parts.append("מידע רלוונטי לשאלה הנוכחית:")
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("chunk_text_only") or chunk.get("text", "")
            if len(content) > self.max_chunk_length:
                content = content[:self.max_chunk_length] + "..."
            
            chunk_line = f"[{i}] "
            if chunk.get("title"):
                chunk_line += f"{chunk['title']} | "
            if include_sources and chunk.get("url"):
                chunk_line += f"{chunk['url']} | "
            chunk_line += content
            
            parts.append(chunk_line)
        
        parts.append("")
        parts.append("שאלה נוכחית:")
        if conversation_history:
            last_user_msg = next(
                (msg["content"] for msg in reversed(conversation_history) if msg.get("role") == "user"),
                ""
            )
            parts.append(last_user_msg)
        
        return "\n".join(parts)
    
    def format_chunk_for_display(self, chunk: Dict, show_full: bool = False) -> str:
        """
        Format a single chunk for display in UI.
        
        Args:
            chunk: Chunk dictionary
            show_full: Whether to show full text or chunk_text_only
        
        Returns:
            Formatted string
        """
        parts = []
        
        if chunk.get("title"):
            parts.append(f"**{chunk['title']}**")
        if chunk.get("subtitle"):
            parts.append(f"*{chunk['subtitle']}*")
        if chunk.get("url"):
            parts.append(f"🔗 {chunk['url']}")
        
        if parts:
            parts.append("")
        
        content = chunk.get("text" if show_full else "chunk_text_only", "")
        parts.append(content)
        
        if chunk.get("score") is not None:
            parts.append(f"\n*רלוונטיות: {chunk['score']:.3f}*")
        
        return "\n".join(parts)


# --- Convenience functions ---

def build_simple_prompt(question: str, chunks: List[Dict], max_chunks: int = 5) -> str:
    """
    Simple convenience function to build a prompt.
    
    Args:
        question: User question
        chunks: Retrieved chunks
        max_chunks: Maximum number of chunks to include
    
    Returns:
        Formatted prompt
    """
    builder = PromptBuilder()
    return builder.build_prompt(question, chunks[:max_chunks])


def build_prompt_with_sources(question: str, chunks: List[Dict], style: str = "detailed") -> str:
    """
    Build prompt with source citations.
    
    Args:
        question: User question
        chunks: Retrieved chunks
        style: Prompt style ("detailed", "concise", "conversational", "structured")
    
    Returns:
        Formatted prompt with sources
    """
    prompt_style = PromptStyle[style.upper()] if hasattr(PromptStyle, style.upper()) else PromptStyle.DETAILED
    builder = PromptBuilder(style=prompt_style, include_metadata=True)
    return builder.build_prompt(question, chunks, include_sources=True)


if __name__ == "__main__":
    # Example usage
    builder = PromptBuilder()
    
    # Sample chunks (as returned by retriever)
    sample_chunks = [
        {
            "id": "doc1::chunk-0",
            "score": 0.85,
            "text": "כותרת: שירות לתושבים\n\nמוקד עירוני 106\nעיריית חיפה מזמינה אתכם לדווח על מפגעים עירוניים.",
            "chunk_text_only": "מוקד עירוני 106\nעיריית חיפה מזמינה אתכם לדווח על מפגעים עירוניים.",
            "url": "https://www.haifa.muni.il/resident-service/service-center/",
            "title": "מוקדי השירות",
            "subtitle": "שירות לתושבים",
        },
        {
            "id": "doc2::chunk-1",
            "score": 0.78,
            "text": "תשלום ארנונה ניתן לבצע דרך האתר או במוקד השירות.",
            "chunk_text_only": "תשלום ארנונה ניתן לבצע דרך האתר או במוקד השירות.",
            "url": "https://www.haifa.muni.il/resident-service/arnona/",
            "title": "תשלום ארנונה",
        },
    ]
    
    question = "איך משלמים ארנונה?"
    
    print("=" * 60)
    print("DETAILED STYLE")
    print("=" * 60)
    prompt = builder.build_prompt(question, sample_chunks)
    print(prompt)
    print("\n" * 2)
    
    print("=" * 60)
    print("CONCISE STYLE")
    print("=" * 60)
    builder.style = PromptStyle.CONCISE
    prompt = builder.build_prompt(question, sample_chunks)
    print(prompt)

