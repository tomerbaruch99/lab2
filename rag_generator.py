"""
RAG Generator with Instruction-Tuned LLM

Uses Google's Gemini API to generate answers from retrieved clauses
with strict formatting policies:
- Always cite clause IDs (filename and category)
- Never fabricate missing information
- Highlight redactions when present
"""

import os
import re
import google.generativeai as genai
from typing import List, Dict, Optional
from consts import GEMINI_MODEL

# Redaction pattern
RE_REDACT = re.compile(r"(\*{2,}|_{2,}|<omitted>)", re.IGNORECASE)

# Default configuration
DEFAULT_MODEL = GEMINI_MODEL
DEFAULT_TEMPERATURE = 0.0  # Deterministic for factual answers


class RAGGenerator:
    """
    Instruction-tuned LLM generator for RAG-based question answering.
    Enforces strict formatting and citation policies.
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        api_key: Optional[str] = None
    ):
        """
        Initialize RAG generator.
        
        Args:
            model: Gemini model name (e.g. 'gemini-1.5-flash')
            temperature: Sampling temperature (0.0 for deterministic)
            api_key: Google API key (uses GEMINI_API_KEY or GOOGLE_API_KEY env var if None)
        """
        self.model_name = model
        self.temperature = temperature
        
        # Get API key
        if api_key:
            api_key_value = api_key
        elif os.getenv("GEMINI_API_KEY"):
            api_key_value = os.getenv("GEMINI_API_KEY")
        elif os.getenv("GOOGLE_API_KEY"):
            api_key_value = os.getenv("GOOGLE_API_KEY")
        else:
            raise ValueError("Google API key not provided. Set GEMINI_API_KEY or GOOGLE_API_KEY env var or pass api_key parameter.")
        
        # Configure Gemini API
        genai.configure(api_key=api_key_value)
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name=model)
    
    def _format_clauses(self, clauses: List[Dict]) -> str:
        """
        Format retrieved clauses with metadata for prompt.
        
        Args:
            clauses: List of clause dictionaries with keys:
                - text/context: Clause text (required)
                - filename: Filename (optional)
                - category: Category name (optional)
                - clause_id: Clause ID (optional)
        
        Returns:
            Formatted string with clause metadata and text
        """
        formatted_clauses = []
        
        for i, clause in enumerate(clauses, 1):
            clause_text = clause.get("text", clause.get("context", ""))
            filename = clause.get("filename", "Unknown")
            category = clause.get("category", "Unknown")
            clause_id = clause.get("clause_id")
            
            # Build clause identifier
            identifier_parts = [f"File: {filename}", f"Category: {category}"]
            if clause_id:
                identifier_parts.append(f"ID: {clause_id}")
            
            identifier = " | ".join(identifier_parts)
            
            # Check for redactions
            has_redaction = bool(RE_REDACT.search(clause_text)) if clause_text else False
            redaction_note = " [CONTAINS REDACTION]" if has_redaction else ""
            
            formatted = f"[Clause {i} - {identifier}]{redaction_note}\n{clause_text}"
            formatted_clauses.append(formatted)
        
        return "\n\n".join(formatted_clauses)
    
    def _build_prompt(
        self,
        question: str,
        clauses: List[Dict],
        include_format_policy: bool = True
    ) -> str:
        """
        Build prompt for instruction-tuned LLM.
        
        Args:
            question: User question
            clauses: List of retrieved clause dictionaries
            include_format_policy: Whether to include formatting instructions
        
        Returns:
            Complete prompt string
        """
        formatted_clauses = self._format_clauses(clauses)
        
        prompt_parts = [
            "You are a contract lawyer AI assistant. Answer the user's question based ONLY on the provided contract clauses.",
            "",
            "RELEVANT CLAUSES:",
            formatted_clauses,
            "",
            f"QUESTION: {question}",
            ""
        ]
        
        if include_format_policy:
            prompt_parts.extend([
                "FORMATTING REQUIREMENTS:",
                "1. Always cite which clause(s) you used by referencing the clause number and identifier (e.g., 'According to Clause 1 - File: contract.pdf | Category: Governing Law').",
                "2. If the answer is not found in the provided clauses, explicitly state 'The answer is not found in the provided clauses.'",
                "3. NEVER fabricate or infer information that is not explicitly stated in the clauses.",
                "4. If a clause contains redaction markers (***, ___, <omitted>), mention this in your answer.",
                "5. For Yes/No questions, provide a clear Yes or No answer with citation.",
                "6. For dates, names, or specific values, quote them exactly as they appear in the clauses.",
                "7. Be concise but complete.",
                ""
            ])
        
        prompt_parts.append("ANSWER:")
        
        return "\n".join(prompt_parts)
    
    def generate(
        self,
        question: str,
        clauses: List[Dict],
        max_tokens: int = 500,
        include_format_policy: bool = True
    ) -> str:
        """
        Generate answer using Gemini API.
        
        Args:
            question: User question
            clauses: List of retrieved clause dictionaries
            max_tokens: Maximum tokens in response (max_output_tokens for Gemini)
            include_format_policy: Whether to include formatting instructions
        
        Returns:
            Generated answer string
        """
        if not clauses:
            return "No relevant clauses were retrieved. Cannot answer the question."
        
        # Build the full prompt with system instruction
        system_instruction = "You are a helpful legal assistant that answers questions about contracts based on provided clauses."
        prompt = self._build_prompt(question, clauses, include_format_policy)
        
        # Combine system instruction with prompt
        full_prompt = f"{system_instruction}\n\n{prompt}"
        
        try:
            # Generate response with generation config
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": max_tokens,
                }
            )
            
            # Extract text from response
            answer = response.text.strip()
            return answer
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def generate_with_metadata(
        self,
        question: str,
        clauses: List[Dict],
        max_tokens: int = 500,
        include_format_policy: bool = True
    ) -> Dict:
        """
        Generate answer with additional metadata.
        
        Args:
            question: User question
            clauses: List of retrieved clause dictionaries
            max_tokens: Maximum tokens in response
            include_format_policy: Whether to include formatting instructions
        
        Returns:
            Dictionary with:
                - answer: Generated answer text
                - num_clauses: Number of clauses used
                - clauses_with_redactions: List of clause indices with redactions
                - model: Model used
        """
        # Check for redactions
        clauses_with_redactions = []
        for i, clause in enumerate(clauses):
            clause_text = clause.get("text", clause.get("context", ""))
            if clause_text and RE_REDACT.search(clause_text):
                clauses_with_redactions.append(i + 1)  # 1-indexed for user display
        
        answer = self.generate(question, clauses, max_tokens, include_format_policy)
        
        return {
            "answer": answer,
            "num_clauses": len(clauses),
            "clauses_with_redactions": clauses_with_redactions,
            "model": self.model_name
        }


def answer_with_rag(
    question: str,
    clauses: List[Dict],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 500
) -> str:
    """
    Convenience function to generate answer using RAG generator.
    
    Args:
        question: User question
        clauses: List of retrieved clause dictionaries
        model: Gemini model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
    
    Returns:
        Generated answer string
    """
    generator = RAGGenerator(model=model, temperature=temperature)
    return generator.generate(question, clauses, max_tokens)

