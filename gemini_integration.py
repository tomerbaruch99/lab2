"""
Gemini Integration for Haifa Municipality RAG
==============================================
Integrates Google Gemini API with the RAG system.

This module:
1. Initializes Gemini models
2. Calls Gemini with prompts from prompt_builder
3. Handles rate limiting and retries
4. Formats responses

Usage:
    from gemini_integration import GeminiRAG
    
    rag = GeminiRAG(api_keys_path="api_keys.json")
    response = rag.answer_question("איך משלמים ארנונה?", top_k=5)
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from retriever import Retriever
from prompt_builder import PromptBuilder, PromptStyle
from utils import DEFAULT_API_KEYS_PATH, DEFAULT_TOP_K


# --- Defaults ---

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_SLEEP_BETWEEN_CALLS = 1.0  # seconds
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_RETRY_DELAY = 10.0  # seconds


# --- Helpers ---

def user_asks_for_evidence(question: str) -> bool:
    """
    Detect if the user is asking for evidence, sources, or further information.
    
    Args:
        question: User's question text
    
    Returns:
        True if the question indicates user wants evidence/detailed sources
    """
    question_lower = question.lower()
    
    # Keywords that suggest user wants evidence or detailed information
    evidence_keywords = [
        "ראיות",  # evidence
        "מקורות",  # sources
        "מסמכים",  # documents
        "תיעוד",  # documentation
        "קובץ מקורי",  # original file
        "מקור",  # source
        "איפה מצאת",  # where did you find
        "איך יודע",  # how do you know
        "הצג לי",  # show me
        "הראה לי",  # show me (another form)
        "פרטים נוספים",  # further details
        "מידע נוסף",  # additional information
        "פרט יותר",  # provide more details
        "pdf",  # explicit PDF request
    ]
    
    return any(keyword in question_lower for keyword in evidence_keywords)


def log_interaction(
    log_file_path: str,
    question: str,
    chunks: List[Dict],
    answer: str,
    has_answer: bool,
) -> None:
    """
    Log an interaction to a JSONL file for traceability.
    
    Args:
        log_file_path: Path to JSONL log file
        question: User's question
        chunks: List of retrieved chunks
        answer: Generated answer or "no relevant info" message
        has_answer: Whether a real answer was returned (False if no chunks found)
    """
    # Format chunks for logging (extract doc_id, url, score)
    logged_chunks = []
    for chunk in chunks:
        logged_chunk = {
            "doc_id": chunk.get("doc_id", ""),
            "url": chunk.get("url", ""),
            "score": chunk.get("score", 0.0),
        }
        # Optionally include more fields if needed
        if chunk.get("title"):
            logged_chunk["title"] = chunk.get("title")
        if chunk.get("file_type"):
            logged_chunk["file_type"] = chunk.get("file_type")
        logged_chunks.append(logged_chunk)
    
    # Create log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "chunks": logged_chunks,
        "num_chunks": len(chunks),
        "has_answer": has_answer,
        "answer": answer,  # Always log the actual answer text
    }
    
    # Append to JSONL file
    # Create directory if it doesn't exist (only if path has a directory component)
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def load_api_keys(api_keys_path: str) -> Dict[str, str]:
    """Load API keys from JSON file."""
    if not os.path.exists(api_keys_path):
        raise FileNotFoundError(f"API keys file not found: {api_keys_path}")
    
    with open(api_keys_path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_gemini(api_keys: Dict[str, str], model_name: str = DEFAULT_GEMINI_MODEL):
    """
    Initialize Gemini model.
    
    Args:
        api_keys: Dictionary with GEMINI_API_KEY
        model_name: Gemini model name (e.g., "gemini-2.5-flash")
    
    Returns:
        Configured GenerativeModel instance
    """
    gemini_api_key = api_keys.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in api_keys.json or environment variables"
        )
    
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name=model_name)
    return model


def call_gemini(
    model,
    prompt: str,
    retry_count: int = 0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_retry_delay: float = DEFAULT_INITIAL_RETRY_DELAY,
    sleep_between_calls: float = DEFAULT_SLEEP_BETWEEN_CALLS,
) -> str:
    """
    Call Gemini API with retry logic for rate limiting.
    
    Args:
        model: Gemini GenerativeModel instance
        prompt: Prompt string
        retry_count: Current retry attempt
        max_retries: Maximum number of retries
        initial_retry_delay: Initial delay before retry (seconds)
        sleep_between_calls: Sleep time between API calls (seconds)
    
    Returns:
        Generated response text
    """
    if retry_count > 0:
        time.sleep(sleep_between_calls)
    
    try:
        response = model.generate_content(prompt)
        # Handle response - Gemini returns text in response.text
        # Use getattr for defensive fallback (same pattern as old-legal_rag)
        return getattr(response, "text", "").strip()
    
    except google_exceptions.ResourceExhausted as e:
        if retry_count >= max_retries:
            raise Exception(f"Rate limit exceeded after {max_retries} retries: {e}")
        
        wait_time = initial_retry_delay * (2 ** retry_count)  # Exponential backoff
        print(f"[WARN] Rate limited. Waiting {wait_time:.1f}s before retry {retry_count + 1}/{max_retries}...")
        time.sleep(wait_time)
        return call_gemini(model, prompt, retry_count + 1, max_retries, initial_retry_delay, sleep_between_calls)
    
    except google_exceptions.InternalServerError as e:
        if retry_count >= max_retries:
            raise Exception(f"Internal server error after {max_retries} retries: {e}")
        
        wait_time = initial_retry_delay * (2 ** retry_count)
        print(f"[WARN] Internal server error. Waiting {wait_time:.1f}s before retry {retry_count + 1}/{max_retries}...")
        time.sleep(wait_time)
        return call_gemini(model, prompt, retry_count + 1, max_retries, initial_retry_delay, sleep_between_calls)
    
    except Exception as e:
        raise Exception(f"Error calling Gemini API: {e}")


# --- Main RAG Class ---

class GeminiRAG:
    """
    Complete RAG system using Gemini for Haifa Municipality.
    
    Combines retrieval, prompt building, and Gemini generation.
    """
    
    def __init__(
        self,
        api_keys_path: str = DEFAULT_API_KEYS_PATH,
        gemini_model_name: str = DEFAULT_GEMINI_MODEL,
        embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        index_name: str = "haifa-municipality-rag-index",
        namespace: Optional[str] = None,
        prompt_style: PromptStyle = PromptStyle.DETAILED,
        sleep_between_calls: float = DEFAULT_SLEEP_BETWEEN_CALLS,
        log_file_path: Optional[str] = None,
    ):
        """
        Initialize the RAG system.
        
        Args:
            api_keys_path: Path to API keys JSON file
            gemini_model_name: Gemini model name
            embedding_model_name: Embedding model for retrieval
            index_name: Pinecone index name
            namespace: Optional namespace
            prompt_style: Prompt style for building prompts
            sleep_between_calls: Sleep time between API calls (rate limiting)
            log_file_path: Optional path to JSONL log file for traceability (e.g., "logs/interactions.jsonl")
        """
        # Load API keys
        self.api_keys = load_api_keys(api_keys_path)
        
        # Initialize Gemini
        print(f"[STEP] Initializing Gemini model '{gemini_model_name}'...")
        self.gemini_model = init_gemini(self.api_keys, gemini_model_name)
        self.gemini_model_name = gemini_model_name
        print(f"[OK] Gemini model ready")
        
        # Initialize retriever
        print(f"[STEP] Initializing retriever...")
        self.retriever = Retriever(
            api_keys_path=api_keys_path,
            embedding_model_name=embedding_model_name,
            index_name=index_name,
            namespace=namespace,
        )
        
        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(style=prompt_style)
        
        self.sleep_between_calls = sleep_between_calls
        self.log_file_path = log_file_path
    
    def answer_question(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        exclude_file_types: Optional[List[str]] = None,
        include_file_types: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        custom_instruction: Optional[str] = None,
        return_chunks: bool = False,
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            exclude_file_types: File types to exclude (e.g., ["pdf"])
            include_file_types: File types to include (e.g., ["html", "txt"])
            filter_dict: Optional metadata filter
            custom_instruction: Custom system instruction
            return_chunks: Whether to return retrieved chunks in response
        
        Returns:
            Dictionary with:
                - answer: Generated answer
                - chunks: Retrieved chunks (if return_chunks=True)
                - prompt: Full prompt sent to Gemini (if return_chunks=True)
        """
        # Determine if we should prefer txt/html or include PDFs equally
        # Prefer txt/html unless user asks for evidence/further information
        # or if explicit file type filters are provided
        asks_for_evidence = user_asks_for_evidence(question)
        prefer_txt_html = not asks_for_evidence and not exclude_file_types and not include_file_types
        
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(
            question,
            top_k=top_k,
            exclude_file_types=exclude_file_types,
            include_file_types=include_file_types,
            filter_dict=filter_dict,
            prefer_txt_html=prefer_txt_html,
        )
        
        # Enforce "no chunks, no answer" pattern - never generate an answer without retrieved context
        if not chunks:
            answer_text = "מצטער, לא מצאתי מידע רלוונטי במאגר המידע של עיריית חיפה."
            
            # Log interaction (no chunks found)
            if self.log_file_path:
                log_interaction(
                    self.log_file_path,
                    question,
                    [],
                    answer_text,
                    has_answer=False,
                )
            
            return {
                "answer": answer_text,
                "chunks": [] if return_chunks else None,
                "prompt": "" if return_chunks else None,
            }
        
        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            question,
            chunks,
            include_sources=True,
            custom_instruction=custom_instruction,
        )
        
        # Call Gemini
        time.sleep(self.sleep_between_calls)
        answer = call_gemini(
            self.gemini_model,
            prompt,
            sleep_between_calls=self.sleep_between_calls,
        )
        
        result = {
            "answer": answer,
        }
        
        if return_chunks:
            result["chunks"] = chunks
            result["prompt"] = prompt
        
        # Log interaction (answer generated)
        if self.log_file_path:
            log_interaction(
                self.log_file_path,
                question,
                chunks,
                answer,
                has_answer=True,
            )
        
        return result
    
    def answer_with_conversation(
        self,
        question: str,
        conversation_history: List[Dict[str, str]],
        top_k: int = DEFAULT_TOP_K,
        exclude_file_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question with conversation history.
        
        Args:
            question: Current user question
            conversation_history: List of {"role": "user"/"assistant", "content": "..."}
            top_k: Number of chunks to retrieve
            exclude_file_types: File types to exclude
        
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve chunks
        chunks = self.retriever.retrieve(
            question,
            top_k=top_k,
            exclude_file_types=exclude_file_types,
        )
        
        # Build chat prompt with history
        prompt = self.prompt_builder.build_chat_prompt(
            conversation_history + [{"role": "user", "content": question}],
            chunks,
        )
        
        # Call Gemini
        time.sleep(self.sleep_between_calls)
        answer = call_gemini(
            self.gemini_model,
            prompt,
            sleep_between_calls=self.sleep_between_calls,
        )
        
        return {
            "answer": answer,
            "chunks": chunks,
        }


# --- CLI ---

def main():
    """CLI for testing Gemini RAG."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Haifa Municipality RAG with Gemini"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask"
    )
    parser.add_argument(
        "--api_keys_path",
        type=str,
        default=DEFAULT_API_KEYS_PATH,
        help="Path to API keys JSON file"
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model name"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--exclude_file_types",
        type=str,
        default=None,
        help="Comma-separated file types to exclude (e.g., 'pdf')"
    )
    parser.add_argument(
        "--show_prompt",
        action="store_true",
        help="Show the full prompt sent to Gemini"
    )
    parser.add_argument(
        "--show_chunks",
        action="store_true",
        help="Show retrieved chunks"
    )
    
    args = parser.parse_args()
    
    # Parse exclude_file_types
    exclude_file_types = None
    if args.exclude_file_types:
        exclude_file_types = [ft.strip() for ft in args.exclude_file_types.split(",")]
    
    # Initialize RAG
    print("[STEP] Initializing RAG system...")
    rag = GeminiRAG(
        api_keys_path=args.api_keys_path,
        gemini_model_name=args.gemini_model,
    )
    
    # Answer question
    print(f"\n[STEP] Answering question...")
    print(f"[QUESTION] {args.question}\n")
    
    result = rag.answer_question(
        args.question,
        top_k=args.top_k,
        exclude_file_types=exclude_file_types,
        return_chunks=args.show_chunks or args.show_prompt,
    )
    
    # Display results
    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result["answer"])
    print()
    
    if args.show_chunks and result.get("chunks"):
        print("=" * 60)
        print("RETRIEVED CHUNKS:")
        print("=" * 60)
        for i, chunk in enumerate(result["chunks"], 1):
            print(f"\n--- Chunk {i} (score: {chunk['score']:.4f}) ---")
            print(f"Title: {chunk.get('title', 'N/A')}")
            print(f"URL: {chunk.get('url', 'N/A')}")
            print(f"Content: {chunk.get('chunk_text_only', '')[:200]}...")
    
    if args.show_prompt and result.get("prompt"):
        print("\n" + "=" * 60)
        print("PROMPT SENT TO GEMINI:")
        print("=" * 60)
        print(result["prompt"])


if __name__ == "__main__":
    main()

