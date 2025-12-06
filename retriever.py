"""
Haifa RAG Retriever
====================================

Retriever supporting:
- Automatic namespace detection from Hebrew query
- Metadata-aware retrieval (doc_type, chunking_strategy)
- Fallback search when namespace returns no results
- Clean, modern architecture matching the new data preparation pipeline
"""

import argparse
from typing import List, Dict, Optional, Any
from pinecone import Pinecone

from utils import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_NAME,
    DEFAULT_API_KEYS_PATH,
    DEFAULT_TOP_K,
    load_pinecone_api_key,
    EmbeddingModel,
)


# ============================================================
# Namespace Detection Rules
# ============================================================

NAMESPACE_RULES = {
    "arnona": ["ארנונה", "תשלום נכס", "חשבון", "מס", "חיוב"],
    "parking": ["חניה", "דוח", "דו\"ח", "קנס", "תווית", "תו"],
    "water": ["מים", "תאגיד", "נזילה", "חשבונית מים"],
    "sanitation": ["זבל", "אשפה", "ניקיון", "תברואה", "פינוי"],
    "welfare": ["רווחה", "שירותים חברתיים", "סיוע", "משפחה"],
    "engineering": ["היתר", "בניין", "הנדסה", "תכנון"],
    "emergency": ["מקלט", "חירום", "אזעקה", "טילים", "מלחמה"],
    "culture": ["אירוע", "תרבות", "מופע", "תערוכה", "חג"],
}

FALLBACK_NAMESPACE = "general"


def detect_namespace(query: str) -> str:
    """Rule-based classifier deciding the correct namespace for a Hebrew query."""

    for ns, keywords in NAMESPACE_RULES.items():
        for kw in keywords:
            if kw in query:
                return ns
    return FALLBACK_NAMESPACE


# ============================================================
# Retriever Class
# ============================================================

class Retriever:
    """
    Updated RAG Retriever for the new Haifa municipality data pipeline.
    """

    def __init__(
        self,
        api_keys_path: str = DEFAULT_API_KEYS_PATH,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        index_name: str = DEFAULT_INDEX_NAME,
    ):
        # Pinecone init
        api_key = load_pinecone_api_key(api_keys_path)
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

        # Embedding model (force CPU to avoid CUDA compatibility issues)
        self.embed_model = EmbeddingModel(embedding_model_name, device="cpu")

        print(f"[INFO] Retriever ready | Index: {index_name}")

    # ------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        strategy: Optional[str] = None,    # "baseline" / "sentence" / "adaptive"
        include_metadata: bool = True,
    ) -> List[Dict]:
        """
        Main retrieval function.
        """

        # 1) Detect namespace
        namespace = detect_namespace(query)
        print(f"[INFO] Query mapped to namespace: {namespace}")

        # 2) Embed query
        q_emb = self.embed_model.embed_query(query)

        # 3) Build filter
        metadata_filter = {}
        if strategy:
            metadata_filter["chunking_strategy"] = strategy

        # 4) Query Pinecone
        results = self.index.query(
            vector=q_emb,
            top_k=top_k,
            include_metadata=include_metadata,
            namespace=namespace,
            filter=metadata_filter if metadata_filter else None,
        )

        # 5) If no results → fallback to general
        if not results.matches:
            print("[WARN] No results in namespace. Falling back to 'general'")
            results = self.index.query(
                vector=q_emb,
                top_k=top_k,
                include_metadata=include_metadata,
                namespace=FALLBACK_NAMESPACE,
                filter=metadata_filter if metadata_filter else None,
            )

        # 6) Format results
        formatted = []
        for m in results.matches:
            formatted.append({
                "id": m.id,
                "score": m.score,
                "metadata": m.metadata,
                "text": m.metadata.get("text", "") if include_metadata else "",
                "chunk_text_only": m.metadata.get("chunk_text_only", ""),
                "url": m.metadata.get("url", ""),
                "title": m.metadata.get("title", ""),
                "subtitle": m.metadata.get("subtitle", ""),
                "doc_id": m.metadata.get("doc_id", ""),
                "chunk_id": m.metadata.get("chunk_id", ""),
                "namespace": m.metadata.get("namespace", ""),
                "chunking_strategy": m.metadata.get("chunking_strategy", ""),
            })

        return formatted


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--strategy", required=False, type=str)
    parser.add_argument("--top_k", default=5, type=int)
    args = parser.parse_args()

    retriever = Retriever()
    results = retriever.retrieve(args.query, top_k=args.top_k, strategy=args.strategy)

    print("\nRESULTS:\n")
    for i, r in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Score: {r['score']:.4f}")
        print(f"Namespace: {r['namespace']}")
        print(f"Strategy: {r['chunking_strategy']}")
        print(f"Title: {r['title']}")
        print(f"URL: {r['url']}")
        print(f"Content: {r['chunk_text_only'][:400]}...\n")


if __name__ == "__main__":
    main()
