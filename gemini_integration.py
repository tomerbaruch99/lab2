"""
Gemini Integration for Haifa Municipality RAG
=============================================

This module wires together:
  - Retriever (Pinecone + embeddings)
  - PromptBuilder
  - Gemini 3 Pro (default: gemini-3-pro-preview)

Changes vs older version:
  - No file_type / PDF filtering logic (we now use doc_type + namespace instead)
  - Uses new Retriever with automatic namespace detection per query
  - Supports chunking_strategy selection (baseline / sentence / adaptive)
  - Uses new metadata schema (namespace, doc_type, chunking_strategy, links)

Example usage:

    from gemini_integration import GeminiRAG

    rag = GeminiRAG(api_keys_path="api_keys.json")
    result = rag.answer_question(
        "איך משלמים ארנונה?",
        top_k=5,
        strategy="adaptive",
        return_chunks=True,
    )
    print(result["answer"])
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
from utils import (
    DEFAULT_API_KEYS_PATH,
    DEFAULT_TOP_K,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_NAME,
    DEFAULT_SLEEP_BETWEEN_CALLS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_INITIAL_RETRY_DELAY,
    EVIDENCE_KEYWORDS,
    EVIDENCE_TOP_K_MULTIPLIER,
    RERANKING_TOP_K_MULTIPLIER,
)
from utils.query_enhancement import rephrase_query, enrich_query, rerank_chunks
from confidence_meter import calculate_confidence


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def user_asks_for_evidence(question: str) -> bool:
    """
    Detect if the user is asking for evidence, sources, or further information.

    Used to decide whether to retrieve more chunks (higher top_k) and
    to encourage the model to expose sources more clearly.
    """
    q = question.lower()
    return any(k in q for k in EVIDENCE_KEYWORDS)


def log_interaction(
    log_file_path: str,
    question: str,
    chunks: List[Dict],
    answer: str,
    has_answer: bool,
) -> None:
    """
    Log an interaction to a JSONL file for traceability.

    The log is intentionally light-weight and schema-stable for later analysis.
    """
    logged_chunks = []
    for ch in chunks:
        logged = {
            "doc_id": ch.get("doc_id", ""),
            "url": ch.get("url", ""),
            "score": ch.get("score", 0.0),
            "namespace": ch.get("namespace", ""),
            "chunking_strategy": ch.get("chunking_strategy", ""),
            "doc_type": ch.get("metadata", {}).get("doc_type", "")
            if isinstance(ch.get("metadata"), dict)
            else "",
        }
        if ch.get("title"):
            logged["title"] = ch["title"]
        logged_chunks.append(logged)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "chunks": logged_chunks,
        "num_chunks": len(chunks),
        "has_answer": has_answer,
        "answer": answer,
    }

    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_api_keys(api_keys_path: str) -> Dict[str, str]:
    """Load API keys from JSON file."""
    if not os.path.exists(api_keys_path):
        raise FileNotFoundError(f"API keys file not found: {api_keys_path}")
    with open(api_keys_path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_gemini(api_keys: Dict[str, str], model_name: str = DEFAULT_GEMINI_MODEL):
    """
    Initialize Gemini GenerativeModel instance.
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
    Call Gemini API with exponential backoff on rate limits / server errors.
    """
    if retry_count > 0:
        time.sleep(sleep_between_calls)

    try:
        response = model.generate_content(prompt)
        return getattr(response, "text", "").strip()

    except google_exceptions.ResourceExhausted as e:
        if retry_count >= max_retries:
            raise Exception(f"Rate limit exceeded after {max_retries} retries: {e}")
        wait = initial_retry_delay * (2 ** retry_count)
        print(f"[WARN] Rate limited. Waiting {wait:.1f}s (retry {retry_count + 1}/{max_retries})")
        time.sleep(wait)
        return call_gemini(
            model,
            prompt,
            retry_count=retry_count + 1,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
            sleep_between_calls=sleep_between_calls,
        )

    except google_exceptions.InternalServerError as e:
        if retry_count >= max_retries:
            raise Exception(f"Internal server error after {max_retries} retries: {e}")
        wait = initial_retry_delay * (2 ** retry_count)
        print(f"[WARN] Internal error. Waiting {wait:.1f}s (retry {retry_count + 1}/{max_retries})")
        time.sleep(wait)
        return call_gemini(
            model,
            prompt,
            retry_count=retry_count + 1,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
            sleep_between_calls=sleep_between_calls,
        )

    except Exception as e:
        raise Exception(f"Error calling Gemini API: {e}")


# ----------------------------------------------------------------------
# Main RAG class
# ----------------------------------------------------------------------

class GeminiRAG:
    """
    Complete RAG system for Haifa Municipality, using the new pipeline:

      - Retriever (with automatic namespace detection)
      - PromptBuilder
      - Gemini 3 Pro (default: gemini-3-pro-preview)
    """

    def __init__(
        self,
        api_keys_path: str = DEFAULT_API_KEYS_PATH,
        gemini_model_name: str = DEFAULT_GEMINI_MODEL,
        embedding_model_name: str = None,
        index_name: str = None,
        prompt_style: PromptStyle = PromptStyle.DETAILED,
        sleep_between_calls: float = DEFAULT_SLEEP_BETWEEN_CALLS,
        log_file_path: Optional[str] = None,
    ):
        """
        Args:
            api_keys_path: Path to api_keys.json with both PINECONE_API_KEY and GEMINI_API_KEY
            gemini_model_name: Gemini model name (defaults to DEFAULT_GEMINI_MODEL)
            embedding_model_name: Embedding model used by the Retriever (defaults to DEFAULT_EMBEDDING_MODEL)
            index_name: Pinecone index name (defaults to DEFAULT_INDEX_NAME)
            prompt_style: Style for PromptBuilder
            sleep_between_calls: Pause between Gemini calls (for safety)
            log_file_path: Optional path for JSONL logs of interactions
        """
        # Use defaults from config if not provided
        if embedding_model_name is None:
            embedding_model_name = DEFAULT_EMBEDDING_MODEL
        if index_name is None:
            index_name = DEFAULT_INDEX_NAME
        
        # API keys
        self.api_keys = load_api_keys(api_keys_path)

        # Gemini
        print(f"[STEP] Initializing Gemini model '{gemini_model_name}'...")
        self.gemini_model = init_gemini(self.api_keys, gemini_model_name)
        self.gemini_model_name = gemini_model_name
        print("[OK] Gemini model ready")

        # Retriever (new version: no namespace argument, namespace decided per query)
        print("[STEP] Initializing Retriever...")
        self.retriever = Retriever(
            api_keys_path=api_keys_path,
            embedding_model_name=embedding_model_name,
            index_name=index_name,
        )
        print("[OK] Retriever ready")

        # Prompt builder
        self.prompt_builder = PromptBuilder(style=prompt_style)

        self.sleep_between_calls = sleep_between_calls
        self.log_file_path = log_file_path

    # ------------------------------------------------------------

    def _determine_effective_top_k(
        self,
        top_k: int,
        question: str,
        use_reranking: bool
    ) -> int:
        """
        Determine the effective top_k based on query characteristics.
        
        Returns a higher top_k if:
        - User asks for evidence (more sources needed)
        - Reranking is enabled (need more candidates to rerank)
        """
        asks_for_evidence = user_asks_for_evidence(question)
        effective_top_k = top_k * EVIDENCE_TOP_K_MULTIPLIER if asks_for_evidence else top_k
        
        if use_reranking:
            effective_top_k = max(effective_top_k, top_k * RERANKING_TOP_K_MULTIPLIER)
        
        return effective_top_k

    def _retrieve_and_rerank_chunks(
        self,
        question: str,
        top_k: int,
        strategy: Optional[str],
        use_reranking: bool
    ) -> List[Dict]:
        """
        Retrieve chunks and optionally rerank them.
        
        Returns:
            List of retrieved chunks (possibly reranked)
        """
        effective_top_k = self._determine_effective_top_k(top_k, question, use_reranking)
        
        # Retrieve
        chunks = self.retriever.retrieve(
            query=question,
            top_k=effective_top_k,
            strategy=strategy,
            include_metadata=True,
        )
        
        # Reranking (optional)
        if use_reranking and chunks:
            chunks = rerank_chunks(question, chunks, top_k, self.gemini_model)
        elif chunks:
            # If not reranking, just take top_k
            chunks = chunks[:top_k]
        
        return chunks

    def _create_no_results_response(self, question: str, return_chunks: bool) -> Dict[str, Any]:
        """Create response when no chunks are retrieved."""
        answer_text = "מצטער, לא מצאתי מידע רלוונטי במאגר המידע של עיריית חיפה."
        
        if self.log_file_path:
            log_interaction(
                self.log_file_path,
                question,
                [],
                answer_text,
                has_answer=False,
            )
        
        confidence_info = {
            "confidence_score": 0.0,
            "confidence_level": "Low",
            "avg_chunk_similarity": 0.0,
            "retrieval_overlap": 0.0,
            "supported_claim_ratio": 0.0,
            "reason": "No relevant chunks retrieved.",
            "unsupported_claims": [],
        }
        
        return {
            "answer": answer_text,
            "confidence": confidence_info,
            "chunks": [] if return_chunks else None,
            "prompt": "" if return_chunks else None,
        }

    def answer_question(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        strategy: Optional[str] = None,       # "baseline" / "sentence" / "adaptive"
        custom_instruction: Optional[str] = None,
        return_chunks: bool = False,
        use_query_enhancement: bool = False,  # Enable query rephrasing/enrichment
        use_reranking: bool = False,           # Enable LLM-based chunk reranking
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        Args:
            question: User question in Hebrew
            top_k: Number of chunks to retrieve
            strategy: Optional chunking_strategy to filter by
                      ("baseline", "sentence", "adaptive"). If None → all.
            custom_instruction: Extra system instruction for PromptBuilder
            return_chunks: If True, also returns chunks and prompt used
            use_query_enhancement: If True, enrich query with additional keywords
            use_reranking: If True, rerank chunks using LLM before generating answer

        Returns:
            {
                "answer": str,
                "confidence": Dict[str, Any],  # Confidence score and details
                "chunks": Optional[List[Dict]],
                "prompt": Optional[str],
                "used_query": Optional[str],  # The actual query used (may be enhanced)
            }
        """
        original_question = question
        
        # Query enhancement (optional)
        if use_query_enhancement:
            question = enrich_query(question, self.gemini_model)
        
        # Retrieve and optionally rerank chunks
        chunks = self._retrieve_and_rerank_chunks(question, top_k, strategy, use_reranking)

        if not chunks:
            return self._create_no_results_response(question, return_chunks)

        # Build prompt from retrieved chunks
        prompt = self.prompt_builder.build_prompt(
            question=question,
            chunks=chunks,
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

        # Calculate confidence score
        confidence_info = calculate_confidence(
            chunks=chunks,
            answer=answer,
            embedding_model=self.retriever.embed_model,
        )

        # Build result
        result = {
            "answer": answer,
            "confidence": confidence_info,
        }
        if return_chunks:
            result["chunks"] = chunks
            result["prompt"] = prompt
        if use_query_enhancement and question != original_question:
            result["used_query"] = question

        # Logging
        if self.log_file_path:
            log_interaction(
                self.log_file_path,
                original_question,
                chunks,
                answer,
                has_answer=True,
            )

        return result

    # ------------------------------------------------------------

    def _format_conversation_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        Format conversation history into text for prompt.
        
        Args:
            conversation_history: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted history text in Hebrew
        """
        if not conversation_history:
            return ""
        
        history_lines = []
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                role_hebrew = "משתמש" if role == "user" else "עוזר"
                history_lines.append(f"{role_hebrew}: {content}")
        
        if history_lines:
            return "הקשר השיחה הקודמת:\n" + "\n".join(history_lines) + "\n\n"
        return ""

    def _build_conversation_instruction(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Build custom instruction combining conversation history with system instruction.
        
        Returns:
            Combined instruction string, or None if no history
        """
        history_text = self._format_conversation_history(conversation_history)
        
        if history_text:
            return history_text + self.prompt_builder.system_instruction
        return None

    def answer_with_conversation(
        self,
        question: str,
        conversation_history: List[Dict[str, str]],
        top_k: int = DEFAULT_TOP_K,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question in the context of a conversation.

        Args:
            question: Current user question
            conversation_history: List of message dicts with 'role' and 'content'
                Example: [
                  {"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."},
                  ...
                ]
            top_k: Number of chunks to retrieve
            strategy: Optional chunking strategy filter
            
        Returns:
            Dictionary with 'answer' and 'chunks'
        """
        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            strategy=strategy,
            include_metadata=True,
        )

        # Build custom instruction with conversation history
        custom_instruction = self._build_conversation_instruction(conversation_history)

        # Build prompt with conversation context
        prompt = self.prompt_builder.build_prompt(
            question=question,
            chunks=chunks,
            include_sources=True,
            custom_instruction=custom_instruction,
        )

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


# ----------------------------------------------------------------------
# CLI for quick testing
# ----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Haifa Municipality RAG with Gemini (updated pipeline)"
    )
    parser.add_argument("--question", type=str, required=True, help="Question to ask")
    parser.add_argument("--api_keys_path", type=str, default=DEFAULT_API_KEYS_PATH)
    parser.add_argument("--gemini_model", type=str, default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Optional chunking strategy: baseline | sentence | adaptive",
    )
    parser.add_argument(
        "--show_prompt",
        action="store_true",
        help="Print the full prompt sent to Gemini",
    )
    parser.add_argument(
        "--show_chunks",
        action="store_true",
        help="Print retrieved chunks",
    )
    parser.add_argument(
        "--use_query_enhancement",
        action="store_true",
        help="Enable query enrichment for better retrieval",
    )
    parser.add_argument(
        "--use_reranking",
        action="store_true",
        help="Enable LLM-based chunk reranking",
    )

    args = parser.parse_args()

    rag = GeminiRAG(
        api_keys_path=args.api_keys_path,
        gemini_model_name=args.gemini_model,
    )

    print(f"[QUESTION] {args.question}")
    if args.strategy:
        print(f"[STRATEGY] {args.strategy}")

    result = rag.answer_question(
        question=args.question,
        top_k=args.top_k,
        strategy=args.strategy,
        return_chunks=args.show_chunks or args.show_prompt,
        use_query_enhancement=args.use_query_enhancement,
        use_reranking=args.use_reranking,
    )

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result["answer"])
    print()

    if args.show_chunks and result.get("chunks"):
        print("=" * 60)
        print("RETRIEVED CHUNKS:")
        print("=" * 60)
        for i, ch in enumerate(result["chunks"], 1):
            print(f"\n--- Chunk {i} (score: {ch['score']:.4f}) ---")
            print(f"Namespace: {ch.get('namespace', '')}")
            print(f"Strategy: {ch.get('chunking_strategy', '')}")
            print(f"Title: {ch.get('title', 'N/A')}")
            print(f"URL: {ch.get('url', 'N/A')}")
            print(f"Content: {ch.get('chunk_text_only', '')[:250]}...")

    if args.show_prompt and result.get("prompt"):
        print("\n" + "=" * 60)
        print("PROMPT SENT TO GEMINI:")
        print("=" * 60)
        print(result["prompt"])


if __name__ == "__main__":
    main()
