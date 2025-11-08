"""
Product 1: Contract Clause Q&A Assistant Pipeline

Unified pipeline that combines:
1. Question → Category mapping (optional)
2. Retrieve top-k clause chunks
3. Extractive reader (for exact spans) + RAG generator (for rationale)
4. Output: Short answer, one-paragraph rationale, citations
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from rag_retriever import retrieve
from extractive_reader import ExtractiveReader
from rag_generator import RAGGenerator
from query_understanding import map_query_to_category, get_category_candidates
from consts import GEMINI_MODEL

# Redaction pattern
RE_REDACT = re.compile(r"(\*{2,}|_{2,}|<omitted>|\[\])", re.IGNORECASE)


class AnswerNormalizer:
    """Normalizes answers to expected formats based on category answer type."""
    
    @staticmethod
    def normalize_date(text: str) -> str:
        """Normalize date formats."""
        # Remove common prefixes
        text = re.sub(r'^(on|as of|effective|dated)\s+', '', text, flags=re.IGNORECASE)
        text = text.strip()
        
        # Try to parse and reformat common date formats
        date_formats = [
            "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
            "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y"
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(text, fmt)
                return dt.strftime("%B %d, %Y")
            except:
                continue
        
        return text  # Return as-is if can't parse
    
    @staticmethod
    def normalize_yesno(text: str) -> str:
        """Normalize Yes/No answers."""
        text_lower = text.lower().strip()
        
        yes_patterns = ["yes", "true", "present", "exists", "included", "required"]
        no_patterns = ["no", "false", "absent", "not present", "not included", "not required"]
        
        if any(pattern in text_lower for pattern in yes_patterns):
            return "Yes"
        elif any(pattern in text_lower for pattern in no_patterns):
            return "No"
        
        return text  # Return as-is if unclear
    
    @staticmethod
    def normalize_duration(text: str) -> str:
        """Normalize duration formats."""
        # Extract common duration patterns
        patterns = [
            (r'(\d+)\s*(?:days?|d)', r'\1 days'),
            (r'(\d+)\s*(?:months?|mo)', r'\1 months'),
            (r'(\d+)\s*(?:years?|yr|y)', r'\1 years'),
            (r'(\d+)\s*(?:weeks?|w)', r'\1 weeks'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    @staticmethod
    def normalize_answer(text: str, answer_type: str) -> str:
        """Normalize answer based on answer type."""
        if not text:
            return ""
        
        if answer_type == "date" or answer_type == "date_or_perpetual":
            return AnswerNormalizer.normalize_date(text)
        elif answer_type == "yesno":
            return AnswerNormalizer.normalize_yesno(text)
        elif answer_type == "duration" or answer_type == "duration_or_perpetual":
            return AnswerNormalizer.normalize_duration(text)
        
        return text.strip()


class QAPipeline:
    """
    Unified Q&A pipeline for contract clause questions.
    Produces: short answer, rationale, citations with span highlights.
    """
    
    def __init__(
        self,
        use_extractive_reader: bool = True,
        use_query_understanding: bool = True,
        extractive_reader_model_dir: str = "models/extractive_reader",
        gemini_model: str = GEMINI_MODEL
    ):
        """
        Initialize Q&A pipeline.
        
        Args:
            use_extractive_reader: Whether to use extractive reader for exact spans
            use_query_understanding: Whether to use query understanding for category mapping
            extractive_reader_model_dir: Path to extractive reader model
            gemini_model: Gemini model name for RAG generator (defaults to GEMINI_MODEL from consts)
        """
        self.use_extractive_reader = use_extractive_reader
        self.use_query_understanding = use_query_understanding
        
        # Initialize extractive reader (optional)
        self.extractive_reader = None
        if use_extractive_reader:
            try:
                self.extractive_reader = ExtractiveReader(model_dir=extractive_reader_model_dir)
            except Exception as e:
                print(f"[WARN] Could not load extractive reader: {e}. Continuing without it.")
                self.use_extractive_reader = False
        
        # Initialize RAG generator
        self.rag_generator = RAGGenerator(model=gemini_model)
        
        # Answer normalizer
        self.normalizer = AnswerNormalizer()
    
    def _format_clause_citation(self, clause: Dict, clause_num: int) -> str:
        """Format clause for citation."""
        filename = clause.get("filename", "Unknown")
        category = clause.get("category", "Unknown")
        clause_id = clause.get("clause_id")
        
        parts = [f"Clause {clause_num}", f"File: {filename}", f"Category: {category}"]
        if clause_id:
            parts.append(f"ID: {clause_id}")
        
        return " | ".join(parts)
    
    def _detect_conflicting_clauses(self, clauses: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect potentially conflicting clauses.
        Returns: (non_conflicting, conflicting)
        """
        # Simple heuristic: if multiple clauses have same category but different answers
        category_clauses = {}
        for clause in clauses:
            category = clause.get("category")
            if category:
                if category not in category_clauses:
                    category_clauses[category] = []
                category_clauses[category].append(clause)
        
        conflicting = []
        non_conflicting = []
        
        for category, cat_clauses in category_clauses.items():
            if len(cat_clauses) > 1:
                # Check if answers differ significantly
                answers = [c.get("answer", "").strip().lower() for c in cat_clauses]
                unique_answers = set(a for a in answers if a)
                
                if len(unique_answers) > 1:
                    conflicting.extend(cat_clauses)
                else:
                    non_conflicting.extend(cat_clauses)
            else:
                non_conflicting.extend(cat_clauses)
        
        return non_conflicting, conflicting
    
    def _handle_missing_clause(
        self,
        question: str,
        category: Optional[str]
    ) -> Dict[str, Any]:
        """Handle case when no relevant clauses are found."""
        # Get related category suggestions
        suggestions = []
        if category:
            suggestions.append(category)
        else:
            # Get top category candidates
            candidates = get_category_candidates(question, top_k=3)
            suggestions = [cat for cat, _ in candidates]
        
        return {
            "short_answer": "Not found in provided clauses",
            "rationale": f"The answer to your question was not found in the provided contract clauses.",
            "citations": [],
            "suggested_categories": suggestions,
            "has_redaction": False,
            "conflicting_clauses": []
        }
    
    def answer(
        self,
        question: str,
        filename: Optional[str] = None,
        k: int = 5,
        use_extractive: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question about contract clauses.
        
        Args:
            question: User's question
            filename: Filter by specific filename (optional)
            k: Number of clauses to retrieve
            use_extractive: Whether to use extractive reader for short answer
        
        Returns:
            Dictionary with:
                - short_answer: Normalized short answer
                - rationale: One-paragraph explanation
                - citations: List of citations with span highlights
                - suggested_categories: Related categories if answer not found
                - has_redaction: Whether any clause contains redactions
                - conflicting_clauses: List of conflicting clauses if any
        """
        # Step 1: Query understanding (optional)
        inferred_category = None
        if self.use_query_understanding:
            inferred_category, confidence = map_query_to_category(question, confidence_threshold=0.3)
            if inferred_category:
                print(f"[Query Understanding] Mapped to category: {inferred_category} (confidence: {confidence:.2f})")
        
        # Step 2: Retrieve top-k clauses
        clauses = retrieve(
            query=question,
            k_dense=8,
            k_bm25=8,
            final_k=k,
            filename=filename,
            category=inferred_category,
            use_query_understanding=self.use_query_understanding
        )
        
        if not clauses:
            return self._handle_missing_clause(question, inferred_category)
        
        # Step 3: Detect conflicts
        non_conflicting, conflicting = self._detect_conflicting_clauses(clauses)
        
        # Step 4: Extract short answer (using extractive reader if available)
        short_answer = ""
        extractive_result = None
        answer_type = "text"  # Default
        
        if use_extractive and self.extractive_reader:
            # Try extractive reader first
            try:
                extractive_result = self.extractive_reader.answer_over_clauses(question, clauses)
                if extractive_result and extractive_result.get("answer_text"):
                    short_answer = extractive_result["answer_text"]
                    # Determine answer type from category
                    if inferred_category:
                        # Look up answer type (simplified - would need full category mapping)
                        if any(term in inferred_category.lower() for term in ["date", "expiration", "effective"]):
                            answer_type = "date"
                        elif any(term in inferred_category.lower() for term in ["yes", "no", "non-", "exclusivity"]):
                            answer_type = "yesno"
                        elif any(term in inferred_category.lower() for term in ["duration", "period", "term"]):
                            answer_type = "duration"
            except Exception as e:
                print(f"[WARN] Error in extractive reader: {e}. Falling back to RAG generator.")
                extractive_result = None
        
        # If extractive reader didn't work, use RAG generator for short answer
        if not short_answer:
            try:
                # Use RAG generator with special prompt for short answer
                rag_result = self.rag_generator.generate(
                    question=question,
                    clauses=clauses,
                    max_tokens=100,  # Short answer only
                    include_format_policy=True
                )
                short_answer = rag_result.split("\n")[0].strip()  # First line
            except Exception as e:
                print(f"[WARN] Error generating short answer: {e}")
                short_answer = "Unable to generate answer"
        
        # Normalize short answer
        short_answer = self.normalizer.normalize_answer(short_answer, answer_type)
        
        # Step 5: Generate rationale using RAG generator
        try:
            rationale = self.rag_generator.generate(
                question=question,
                clauses=clauses,
                max_tokens=300,
                include_format_policy=True
            )
        except Exception as e:
            print(f"[WARN] Error generating rationale: {e}")
            rationale = f"Error generating rationale: {str(e)}"
        
        # Step 6: Build citations with span highlights
        citations = []
        has_redaction = False
        
        for i, clause in enumerate(clauses, 1):
            clause_text = clause.get("text", clause.get("context", ""))
            
            # Check for redactions
            clause_has_redaction = bool(RE_REDACT.search(clause_text)) if clause_text else False
            if clause_has_redaction:
                has_redaction = True
            
            # Get span indices if extractive result matches this clause
            start_idx = None
            end_idx = None
            
            if extractive_result:
                if (clause.get("filename") == extractive_result.get("filename") and
                    clause.get("category") == extractive_result.get("category")):
                    start_idx = extractive_result.get("start")
                    end_idx = extractive_result.get("end")
            
            citation = {
                "clause_num": i,
                "citation": self._format_clause_citation(clause, i),
                "text": clause_text,
                "filename": clause.get("filename"),
                "category": clause.get("category"),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "has_redaction": clause_has_redaction
            }
            citations.append(citation)
        
        # Step 7: Handle conflicting clauses
        conflicting_info = []
        if conflicting:
            for conf_clause in conflicting:
                conflicting_info.append({
                    "citation": self._format_clause_citation(conf_clause, clauses.index(conf_clause) + 1),
                    "text": conf_clause.get("text", conf_clause.get("context", "")),
                    "category": conf_clause.get("category")
                })
        
        return {
            "short_answer": short_answer,
            "rationale": rationale,
            "citations": citations,
            "suggested_categories": [] if clauses else get_category_candidates(question, top_k=3),
            "has_redaction": has_redaction,
            "conflicting_clauses": conflicting_info,
            "extractive_span": {
                "start": extractive_result.get("start") if extractive_result else None,
                "end": extractive_result.get("end") if extractive_result else None,
                "text": extractive_result.get("answer_text") if extractive_result else None
            } if extractive_result else None
        }


def answer_question(
    question: str,
    filename: Optional[str] = None,
    k: int = 5,
    use_extractive: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to answer a question.
    
    Args:
        question: User's question
        filename: Filter by specific filename (optional)
        k: Number of clauses to retrieve
        use_extractive: Whether to use extractive reader
    
    Returns:
        Answer dictionary with short_answer, rationale, citations, etc.
    """
    pipeline = QAPipeline()
    return pipeline.answer(question, filename, k, use_extractive)

