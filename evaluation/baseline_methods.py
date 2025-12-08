"""
Baseline Methods for RAG Evaluation
====================================

This module implements baseline retrieval methods for comparison:
1. TF-IDF keyword-based retrieval
2. Retrieval-only baseline (semantic search without generation)
3. Simple keyword matching baseline

These baselines are used to contextualize the performance of the full RAG system
and demonstrate improvements over traditional retrieval approaches.
"""

import json
import re
import sys
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARN] scikit-learn not available. TF-IDF baseline will be disabled.")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from retriever import Retriever, detect_namespace, FALLBACK_NAMESPACE
from utils import DEFAULT_API_KEYS_PATH, DEFAULT_TOP_K, DEFAULT_EMBEDDING_MODEL


# ============================================================
# TF-IDF Baseline Retriever
# ============================================================

class TfIdfBaselineRetriever:
    """
    TF-IDF based keyword retrieval baseline.
    
    This baseline uses traditional TF-IDF vectorization to rank documents
    by keyword relevance, similar to classic search engines.
    
    References:
    - Salton & Buckley (1988): Term-weighting approaches in automatic text retrieval
    - Manning, Raghavan & Schütze (2008): Introduction to Information Retrieval
    """
    
    def __init__(
        self,
        data_file: str = "scrape_and_prepare_data/haifa_prepared_data/haifa_rag_chunks.parquet",
        api_keys_path: str = DEFAULT_API_KEYS_PATH,
    ):
        """
        Initialize TF-IDF retriever.
        
        Args:
            data_file: Path to parquet file with chunked documents
            api_keys_path: Path to API keys (used for namespace detection)
        """
        self.api_keys_path = api_keys_path
        
        # Check if sklearn is available
        if not SKLEARN_AVAILABLE:
            self.documents = []
            self.vectorizer = None
            self.tfidf_matrix = None
            print("[WARN] scikit-learn not available. TF-IDF baseline disabled.")
            return
        
        # Load data if file exists (path relative to current working directory - main project directory)
        data_path = Path(data_file)
        
        if data_path.exists():
            df = pd.read_parquet(data_path)
            self.documents = df.to_dict('records')
            
            # Build TF-IDF index
            self._build_tfidf_index()
        else:
            self.documents = []
            self.vectorizer = None
            self.tfidf_matrix = None
            print(f"[WARN] Data file not found: {data_path}. TF-IDF baseline unavailable.")
    
    def _preprocess_text(self, text: str) -> str:
        """Simple Hebrew text preprocessing."""
        if not isinstance(text, str):
            return ""
        # Remove special characters, keep Hebrew and alphanumeric
        text = re.sub(r'[^\w\s\u0590-\u05FF]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower()
    
    def _build_tfidf_index(self):
        """Build TF-IDF index from loaded documents."""
        if not self.documents:
            return
        
        print("[INFO] Building TF-IDF index...")
        
        # Extract text for each document
        texts = []
        for doc in self.documents:
            # Combine title, subtitle, and chunk text
            text_parts = []
            if doc.get('title'):
                text_parts.append(str(doc['title']))
            if doc.get('subtitle'):
                text_parts.append(str(doc['subtitle']))
            if doc.get('chunk_text_only'):
                text_parts.append(str(doc['chunk_text_only']))
            elif doc.get('text'):
                text_parts.append(str(doc['text']))
            
            combined_text = ' '.join(text_parts)
            texts.append(self._preprocess_text(combined_text))
        
        # Build TF-IDF vectorizer
        # Using Hebrew-friendly tokenization (split by whitespace)
        self.vectorizer = TfidfVectorizer(
            token_pattern=r'\S+',  # Match any non-whitespace sequence
            max_features=10000,
            min_df=2,  # Ignore terms that appear in < 2 documents
            max_df=0.95,  # Ignore terms that appear in > 95% of documents
            ngram_range=(1, 2),  # Use unigrams and bigrams
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"[OK] TF-IDF index built: {len(self.documents)} documents, {self.tfidf_matrix.shape[1]} features")
    
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        strategy: Optional[str] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
    ) -> List[Dict]:
        """
        Retrieve documents using TF-IDF (Term Frequency-Inverse Document Frequency) similarity.
        
        This method uses traditional keyword-based retrieval, computing TF-IDF vectors
        for the query and all documents, then ranking by cosine similarity. This provides
        a baseline comparison against semantic embedding-based retrieval.
        
        Args:
            query: Query text in Hebrew (will be preprocessed)
            top_k: Number of top results to return
            strategy: Optional chunking strategy filter ("baseline", "sentence", "adaptive")
                     for compatibility with main retriever interface
            namespace: Optional namespace filter (e.g., "arnona", "parking")
                      If None, automatically detects namespace from query
            include_metadata: Whether to include full metadata in results
            
        Returns:
            List of chunk dictionaries, each containing:
            - id: Document-chunk identifier
            - score: TF-IDF cosine similarity score (0-1)
            - metadata: Full document metadata (if include_metadata=True)
            - chunk_text_only: Chunk content text
            - url, title, subtitle, doc_id, chunk_id, namespace, chunking_strategy
            
        Process:
            1. Preprocess query (normalize, lowercase, remove special chars)
            2. Transform query to TF-IDF vector
            3. Compute cosine similarity with all document vectors
            4. Get top-k indices sorted by similarity
            5. Filter by strategy and namespace if specified
            6. Format results to match Retriever interface
            
        Note:
            - Returns empty list if documents not loaded or sklearn unavailable
            - Retrieves 2x top_k initially, then filters to top_k
            - Scores are TF-IDF cosine similarities (typically 0-1 range)
            - Results are sorted by similarity (highest first)
        """
        if not self.documents or self.vectorizer is None:
            return []
        
        # Detect namespace if not provided
        if namespace is None:
            namespace = detect_namespace(query)
        
        # Preprocess query
        query_processed = self._preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query_processed])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get extra for filtering
        
        # Filter and format results
        results = []
        for idx in top_indices:
            if similarities[idx] <= 0:
                continue
            
            doc = self.documents[idx]
            
            # Apply filters
            if strategy and doc.get('chunking_strategy') != strategy:
                continue
            if namespace and doc.get('namespace') != namespace:
                continue
            
            # Format result (compatible with Retriever format)
            result = {
                "id": str(doc.get('doc_id', '')) + '_' + str(doc.get('chunk_id', '')),
                "score": float(similarities[idx]),
                "metadata": doc if include_metadata else {},
                "text": doc.get('text', '') if include_metadata else '',
                "chunk_text_only": doc.get('chunk_text_only', ''),
                "url": doc.get('url', ''),
                "title": doc.get('title', ''),
                "subtitle": doc.get('subtitle', ''),
                "doc_id": doc.get('doc_id', ''),
                "chunk_id": doc.get('chunk_id', ''),
                "namespace": doc.get('namespace', ''),
                "chunking_strategy": doc.get('chunking_strategy', ''),
            }
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results


# ============================================================
# Retrieval-Only Baseline (No Generation)
# ============================================================

class RetrievalOnlyBaseline:
    """
    Retrieval-only baseline that uses semantic search but returns
    retrieved chunks directly without LLM generation.
    
    This baseline demonstrates the contribution of the generation component
    by showing performance with retrieval alone.
    
    Reference:
    - Lewis et al. (2020): Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
    """
    
    def __init__(
        self,
        api_keys_path: str = DEFAULT_API_KEYS_PATH,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        index_name: str = None,
    ):
        """
        Initialize retrieval-only baseline.
        
        Args:
            api_keys_path: Path to API keys file
            embedding_model_name: Embedding model name (defaults to DEFAULT_EMBEDDING_MODEL)
            index_name: Pinecone index name (defaults to DEFAULT_INDEX_NAME)
        """
        if index_name is None:
            index_name = DEFAULT_INDEX_NAME
        self.retriever = Retriever(
            api_keys_path=api_keys_path,
            embedding_model_name=embedding_model_name,
            index_name=index_name,
        )
    
    def answer_question(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Answer question by returning top retrieved chunks directly (no LLM generation).
        
        This baseline demonstrates the contribution of the generation component
        by showing what happens when we skip LLM generation and just return
        retrieved chunks. Useful for understanding how much value the LLM adds.
        
        Args:
            question: User question in Hebrew
            top_k: Number of chunks to retrieve and concatenate
            strategy: Optional chunking strategy filter ("baseline", "sentence", "adaptive")
            
        Returns:
            Dictionary containing:
            - answer: Concatenated text from top_k retrieved chunks
            - chunks: List of retrieved chunk dictionaries
            - method: "retrieval_only" identifier
            
        Note:
            - No LLM is used - just direct retrieval and concatenation
            - Answer quality depends entirely on retrieval quality
            - Useful for comparing against full RAG system to measure LLM contribution
        """
        # Retrieve chunks
        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            strategy=strategy,
            include_metadata=True,
        )
        
        if not chunks:
            return {
                "answer": "מצטער, לא מצאתי מידע רלוונטי במאגר המידע.",
                "chunks": chunks,
                "method": "retrieval_only",
            }
        
        # Concatenate top chunks as answer
        answer_parts = []
        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk.get('chunk_text_only') or chunk.get('text', '')
            if chunk_text:
                answer_parts.append(f"[{i}] {chunk_text}")
        
        answer = "\n\n".join(answer_parts)
        
        return {
            "answer": answer,
            "chunks": chunks,
            "method": "retrieval_only",
        }


# ============================================================
# Simple Keyword Matching Baseline
# ============================================================

class KeywordMatchingBaseline:
    """
    Simple keyword matching baseline for comparison.
    
    This baseline uses exact keyword matching, similar to basic
    search engines or database queries.
    """
    
    def __init__(
        self,
        data_file: str = "scrape_and_prepare_data/haifa_prepared_data/haifa_rag_chunks.parquet",
    ):
        """Initialize keyword matching baseline."""
        data_path = Path(data_file)
        if not data_path.is_absolute():
            # Try relative to parent directory
            data_path = Path(__file__).parent.parent / data_file
        
        if data_path.exists():
            df = pd.read_parquet(data_path)
            self.documents = df.to_dict('records')
        else:
            self.documents = []
            print(f"[WARN] Data file not found: {data_path}. Keyword baseline unavailable.")
    
    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text."""
        if not isinstance(text, str):
            return set()
        # Simple tokenization
        words = re.findall(r'\S+', text.lower())
        # Filter out very short words
        keywords = {w for w in words if len(w) > 2}
        return keywords
    
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        strategy: Optional[str] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
    ) -> List[Dict]:
        """
        Retrieve documents using keyword matching.
        
        Args:
            query: Query text
            top_k: Number of results to return
            strategy: Optional chunking strategy filter
            namespace: Optional namespace filter
            
        Returns:
            List of retrieved chunks with match scores
        """
        if not self.documents:
            return []
        
        # Extract query keywords
        query_keywords = self._extract_keywords(query)
        
        if not query_keywords:
            return []
        
        # Detect namespace if not provided
        if namespace is None:
            namespace = detect_namespace(query)
        
        # Score documents by keyword overlap
        scored_docs = []
        for doc in self.documents:
            # Apply filters
            if strategy and doc.get('chunking_strategy') != strategy:
                continue
            if namespace and doc.get('namespace') != namespace:
                continue
            
            # Combine document text
            text_parts = []
            if doc.get('title'):
                text_parts.append(str(doc['title']))
            if doc.get('subtitle'):
                text_parts.append(str(doc['subtitle']))
            if doc.get('chunk_text_only'):
                text_parts.append(str(doc['chunk_text_only']))
            elif doc.get('text'):
                text_parts.append(str(doc['text']))
            
            doc_text = ' '.join(text_parts)
            doc_keywords = self._extract_keywords(doc_text)
            
            # Compute Jaccard similarity (intersection over union)
            intersection = len(query_keywords & doc_keywords)
            union = len(query_keywords | doc_keywords)
            score = intersection / union if union > 0 else 0.0
            
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and return top-k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, doc in scored_docs[:top_k]:
            result = {
                "id": str(doc.get('doc_id', '')) + '_' + str(doc.get('chunk_id', '')),
                "score": float(score),
                "metadata": doc if include_metadata else {},
                "text": doc.get('text', '') if include_metadata else '',
                "chunk_text_only": doc.get('chunk_text_only', ''),
                "url": doc.get('url', ''),
                "title": doc.get('title', ''),
                "subtitle": doc.get('subtitle', ''),
                "doc_id": doc.get('doc_id', ''),
                "chunk_id": doc.get('chunk_id', ''),
                "namespace": doc.get('namespace', ''),
                "chunking_strategy": doc.get('chunking_strategy', ''),
            }
            results.append(result)
        
        return results

