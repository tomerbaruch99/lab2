"""
Compare different embedding models for Hebrew text retrieval.

This script provides two comparison modes:

1. DIRECT SIMILARITY TEST (Recommended)
   - Tests embedding quality directly without Pinecone index
   - Compares how well each model captures semantic similarity for Hebrew text
   - No index needed - works with any documents
   - Usage: python utils/compare_embedding_models.py --similarity_test

2. RETRIEVAL COMPARISON (Requires matching indexes)
   - Tests retrieval quality using Pinecone index
   - ⚠️  WARNING: Index must be created with the same embedding model
   - Using different models will give incorrect/incomparable results
   - Usage: python utils/compare_embedding_models.py [other options]

All models tested are SentenceTransformer models from the sentence-transformers library.

SentenceTransformer models tested:
- all-MiniLM-L6-v2 (default, fast)
- paraphrase-multilingual-MiniLM-L12-v2 (better Hebrew, recommended)
- intfloat/multilingual-e5-large (high quality, used in Municipality-RAG, recommended)
- paraphrase-multilingual-mpnet-base-v2 (high quality, slower)
- distiluse-base-multilingual-cased-v2 (balanced)
- intfloat/multilingual-e5-base (smaller E5 variant)
- intfloat/multilingual-e5-small (same family as e5-base, smaller & faster)
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (multilingual MiniLM variant, faster than e5-base)
- sentence-transformers/all-mpnet-base-v2 (English baseline)
"""

import sys
import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple

try:
    import yaml  # optional, for YAML test sets
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Add parent directory to path to import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import embedding model directly from file (avoids Pinecone dependencies)
import importlib.util
embedding_path = os.path.join(project_root, "utils", "embedding.py")
spec = importlib.util.spec_from_file_location("embedding_module", embedding_path)
embedding_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embedding_module)
EmbeddingModel = embedding_module.EmbeddingModel

# Optional imports for retrieval comparison
PINECONE_AVAILABLE = False

try:
    from retriever import Retriever
    from utils import DEFAULT_INDEX_NAME, DEFAULT_API_KEYS_PATH
    PINECONE_AVAILABLE = True
except ImportError:
    # Pinecone not available - similarity test can still run
    DEFAULT_INDEX_NAME = None
    DEFAULT_API_KEYS_PATH = None
    pass


# SentenceTransformer models to compare
# All models are from the sentence-transformers library
EMBEDDING_MODELS = [
    {
        "name": "all-MiniLM-L6-v2",
        "description": "SentenceTransformer - Default, fast, multilingual, good general performance",
        "recommended": False,
        "type": "SentenceTransformer",
    },
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "description": "SentenceTransformer - Better Hebrew support, larger, multilingual",
        "recommended": True,
        "type": "SentenceTransformer",
    },
    {
        "name": "intfloat/multilingual-e5-large",
        "description": "SentenceTransformer - Best for multilingual (used in Municipality-RAG), large, high quality",
        "recommended": True,
        "type": "SentenceTransformer",
    },
    {
        "name": "paraphrase-multilingual-mpnet-base-v2",
        "description": "SentenceTransformer - High quality multilingual, larger, slower",
        "recommended": False,
        "type": "SentenceTransformer",
    },
    {
        "name": "distiluse-base-multilingual-cased-v2",
        "description": "SentenceTransformer - Multilingual USE model, good balance",
        "recommended": False,
        "type": "SentenceTransformer",
    },
    {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "description": "SentenceTransformer - Explicit path version of multilingual MiniLM",
        "recommended": False,
        "type": "SentenceTransformer",
    },
    {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "description": "SentenceTransformer - English-optimized, high quality (baseline)",
        "recommended": False,
        "type": "SentenceTransformer",
    },
    {
        "name": "intfloat/multilingual-e5-base",
        "description": "SentenceTransformer - Multilingual E5 base (smaller than large)",
        "recommended": False,
        "type": "SentenceTransformer",
    },
    {
        "name": "intfloat/multilingual-e5-small",
        "description": "SentenceTransformer - Multilingual E5 small (same family as e5-base, smaller & faster)",
        "recommended": False,
        "type": "SentenceTransformer",
    },
]


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def test_model_with_index(
    model_name: str,
    query: str,
    index_name: str,
    api_keys_path: str,
    top_k: int = 5,
) -> Dict:
    """
    Test a single embedding model using Pinecone index.
    
    WARNING: This only works correctly if the index was created with the same model.
    Using a different model will give incorrect/incomparable results.
    
    Note: File type filtering is no longer supported. Use strategy parameter in retriever instead.
    """
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    print("⚠️  WARNING: Results only valid if index was created with this model!")
    
    try:
        retriever = Retriever(
            api_keys_path=api_keys_path,
            embedding_model_name=model_name,
            index_name=index_name,
        )
        
        # Retrieve results
        # Note: File type filtering is no longer supported, using strategy filter instead
        results = retriever.retrieve(
            query=query,
            top_k=top_k,
            strategy=None,  # Can use "baseline", "sentence", or "adaptive" if needed
            include_metadata=True,
        )
        
        # Analyze results
        if not results:
            return {
                "model": model_name,
                "success": False,
                "error": "No results found",
                "results": [],
            }
        
        # Calculate statistics
        avg_score = sum(r["score"] for r in results) / len(results)
        max_score = max(r["score"] for r in results)
        min_score = min(r["score"] for r in results)
        
        doc_types = {}
        for r in results:
            dt = r.get("doc_type", "unknown")
            doc_types[dt] = doc_types.get(dt, 0) + 1
        
        # Check for generic PDF titles
        generic_titles = sum(
            1 for r in results 
            if r.get("title", "").lower() == "pdf document"
        )
        
        return {
            "model": model_name,
            "success": True,
            "num_results": len(results),
            "avg_score": avg_score,
            "max_score": max_score,
            "min_score": min_score,
            "doc_types": doc_types,
            "generic_titles": generic_titles,
            "results": results,
        }
        
    except Exception as e:
        return {
            "model": model_name,
            "success": False,
            "error": str(e),
            "results": [],
        }


def load_testset(path: str) -> List[Dict[str, Any]]:
    """
    Load a test set from JSON or YAML.

    Expected format (JSON/YAML):

    - Either:
        { "queries": [ { "query": "...", "documents": [...] }, ... ] }
      Or:
        [ { "query": "...", "documents": [...] }, ... ]

    Each document:
        {
          "text": "document text",
          "label": "relevant" | "irrelevant" | ...
          or
          "is_relevant": true/false
        }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Testset file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    with open(path, "r", encoding="utf-8") as f:
        if ext in [".json"]:
            data = json.load(f)
        elif ext in [".yml", ".yaml"]:
            if not YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is not installed. Install it with 'pip install pyyaml' "
                    "or use a JSON test set."
                )
            data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported testset extension: {ext}. Use .json or .yaml")

    # Normalize: data can be {"queries": [...]} or just [...]
    if isinstance(data, dict) and "queries" in data:
        queries = data["queries"]
    else:
        queries = data

    if not isinstance(queries, list):
        raise ValueError("Testset must be a list of queries or an object with 'queries' list")

    # Basic validation
    for i, q in enumerate(queries):
        if "query" not in q or "documents" not in q:
            raise ValueError(
                f"Each entry must have 'query' and 'documents' keys. Problem at index {i}."
            )
        if not isinstance(q["documents"], list):
            raise ValueError(f"'documents' must be a list (query index {i}).")

    return queries


def is_doc_relevant(doc: Dict[str, Any]) -> bool:
    """Determine if a document is labeled as relevant."""
    # Boolean flag takes priority if present
    if "is_relevant" in doc:
        return bool(doc["is_relevant"])

    label = str(doc.get("label", "")).strip().lower()
    relevant_labels = {"relevant", "pos", "positive", "gold", "true"}
    return label in relevant_labels


def test_embedding_similarity(
    model_name: str,
    query: str,
    test_documents: List[str],
) -> Dict:
    """
    Test embedding similarity directly without using Pinecone.
    
    This tests how well each model captures semantic similarity for Hebrew text.
    """
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} (Direct similarity)")
    print(f"{'='*60}")
    
    try:
        # Load embedding model (force CPU to avoid CUDA compatibility issues)
        embed_model = EmbeddingModel(model_name, device="cpu", verbose=False)
        
        # Embed query
        query_embedding = embed_model.embed_query(query)
        
        # Embed all test documents
        doc_embeddings = embed_model.embed(test_documents)
        
        # Calculate similarities
        similarities = []
        for doc_text, doc_emb in zip(test_documents, doc_embeddings):
            sim = cosine_similarity(query_embedding, doc_emb)
            similarities.append(sim)
        
        # Analyze results
        avg_sim = np.mean(similarities)
        max_sim = np.max(similarities)
        min_sim = np.min(similarities)
        std_sim = np.std(similarities)
        
        # Sort by similarity
        sorted_pairs = sorted(zip(test_documents, similarities), key=lambda x: x[1], reverse=True)
        
        return {
            "model": model_name,
            "success": True,
            "avg_similarity": float(avg_sim),
            "max_similarity": float(max_sim),
            "min_similarity": float(min_sim),
            "std_similarity": float(std_sim),
            "top_matches": sorted_pairs[:3],
        }
        
    except Exception as e:
        import traceback
        return {
            "model": model_name,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def print_model_results(model_result: Dict, show_details: bool = True):
    """Print results for a single model."""
    if not model_result["success"]:
        print(f"❌ FAILED: {model_result.get('error', 'Unknown error')}")
        return
    
    print(f"\n✅ Success")
    print(f"   Results: {model_result['num_results']}")
    print(f"   Average score: {model_result['avg_score']:.4f}")
    print(f"   Score range: {model_result['min_score']:.4f} - {model_result['max_score']:.4f}")
    print(f"   Document types: {model_result['doc_types']}")
    if model_result['generic_titles'] > 0:
        print(f"   ⚠️  Generic titles: {model_result['generic_titles']}")
    
    if show_details and model_result['results']:
        print(f"\n   Top {min(3, len(model_result['results']))} results:")
        for i, result in enumerate(model_result['results'][:3], 1):
            print(f"     {i}. Score: {result['score']:.4f} | "
                  f"Type: {result.get('doc_type', 'unknown')} | "
                  f"Title: {result.get('title', 'N/A')[:50]}")


def compare_similarity_direct(
    query: str,
    test_documents: List[str] = None,
):
    """
    Compare embedding models using direct similarity (no Pinecone index needed).
    
    This tests how well each model captures semantic similarity for Hebrew text.
    """
    if test_documents is None:
        # Default test documents relevant to the query
        test_documents = [
            "תשלום ארנונה ניתן לבצע דרך אתר העירייה או במוקד השירות",
            "ארנונה היא תשלום חובה לתושבי העיר עבור שירותים עירוניים",
            "ניתן לשלם ארנונה באופן מקוון, במוקד השירות או באמצעות העברה בנקאית",
            "המוקד העירוני מעניק שירות לתושבים בנושאי ארנונה ושירותים נוספים",
            "שעות פעילות המוקד העירוני הן 24 שעות ביממה",
        ]
    
    print("="*60)
    print("DIRECT EMBEDDING SIMILARITY COMPARISON")
    print("="*60)
    print(f"\nAll models: SentenceTransformer (sentence-transformers library)")
    print(f"Query: {query}")
    print(f"Test documents: {len(test_documents)}")
    print("\n⚠️  NOTE: This tests embedding quality directly, no index needed.")
    print("    For retrieval comparison, you need indexes created with each model.\n")
    
    results = []
    
    for model_info in EMBEDDING_MODELS:
        model_name = model_info["name"]
        print(f"\n{'='*60}")
        print(f"SentenceTransformer Model: {model_name}")
        if model_info.get("recommended"):
            print("⭐ RECOMMENDED")
        print(f"Description: {model_info['description']}")
        
        result = test_embedding_similarity(model_name, query, test_documents)
        results.append(result)
        
        if result["success"]:
            print(f"✅ Success (SentenceTransformer)")
            print(f"   Average similarity: {result['avg_similarity']:.4f}")
            print(f"   Max similarity: {result['max_similarity']:.4f}")
            print(f"   Std deviation: {result['std_similarity']:.4f}")
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
    
    # Summary
    successful = [r for r in results if r["success"]]
    if successful:
        successful.sort(key=lambda x: x["avg_similarity"], reverse=True)
        print(f"\n{'='*60}")
        print("RANKED BY AVERAGE SIMILARITY")
        print("="*60)
        for i, result in enumerate(successful, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"{medal} {result['model']}: {result['avg_similarity']:.4f} (max: {result['max_similarity']:.4f})")


def evaluate_models_on_testset(
    test_queries: List[Dict[str, Any]],
    top_k: int = 3,
) -> None:
    """
    Evaluate all embedding models on a labeled test set.

    Metrics per model:
    - top-1 hit rate: fraction of queries where the top-ranked doc is relevant
    - top-k hit rate: fraction of queries where any of the top-k docs is relevant
    - mean similarity for relevant vs. irrelevant documents
    """
    print("=" * 60)
    print("EMBEDDING MODEL EVALUATION ON LABELED TEST SET")
    print("=" * 60)
    print(f"Queries in test set: {len(test_queries)}")
    print(f"Top-k for hit rate: {top_k}\n")

    # Pre-extract just the text for each query to avoid re-parsing
    all_queries_texts: List[str] = [q["query"] for q in test_queries]
    all_docs_texts: List[List[str]] = [
        [d["text"] for d in q["documents"]] for q in test_queries
    ]
    all_relevance_masks: List[List[bool]] = [
        [is_doc_relevant(d) for d in q["documents"]] for q in test_queries
    ]

    for model_info in EMBEDDING_MODELS:
        model_name = model_info["name"]
        print("\n" + "=" * 60)
        print(f"Model: {model_name}")
        if model_info.get("recommended"):
            print("⭐ RECOMMENDED")
        print(f"Description: {model_info['description']}")

        try:
            # Force CPU to avoid CUDA compatibility issues
            embed_model = EmbeddingModel(model_name, device="cpu", verbose=False)
        except Exception as e:
            print(f"❌ FAILED to load model: {e}")
            continue

        num_queries = 0
        top1_hits = 0
        topk_hits = 0

        relevant_sims_all: List[float] = []
        irrelevant_sims_all: List[float] = []

        for q_text, docs_texts, relevance_mask in zip(
            all_queries_texts, all_docs_texts, all_relevance_masks
        ):
            # Skip queries with no labeled relevant docs
            if not any(relevance_mask):
                continue

            num_queries += 1

            # Embed query and documents
            try:
                q_vec = embed_model.embed_query(q_text)
                d_vecs = embed_model.embed(docs_texts)
            except Exception as e:
                print(f"   ⚠️ Error embedding a query: {e}")
                continue

            # Compute similarities
            sims = [
                cosine_similarity(q_vec, d_vec) for d_vec in d_vecs
            ]

            # Collect sims for analysis
            for sim, is_rel in zip(sims, relevance_mask):
                if is_rel:
                    relevant_sims_all.append(sim)
                else:
                    irrelevant_sims_all.append(sim)

            # Rank documents by similarity
            ranked_indices = sorted(
                range(len(sims)), key=lambda idx: sims[idx], reverse=True
            )

            # Top-1 hit
            top1_idx = ranked_indices[0]
            if relevance_mask[top1_idx]:
                top1_hits += 1

            # Top-k hit
            topk_indices = ranked_indices[:top_k]
            if any(relevance_mask[idx] for idx in topk_indices):
                topk_hits += 1

        if num_queries == 0:
            print("❌ No queries with at least one relevant document; cannot compute metrics.")
            continue

        # Aggregate metrics
        top1_hit_rate = top1_hits / num_queries
        topk_hit_rate = topk_hits / num_queries

        rel_mean = float(np.mean(relevant_sims_all)) if relevant_sims_all else float("nan")
        irrel_mean = float(np.mean(irrelevant_sims_all)) if irrelevant_sims_all else float("nan")

        print(f"\n✅ Evaluation complete on {num_queries} queries with labeled relevant docs.")
        print(f"   Top-1 hit rate: {top1_hit_rate:.3f}")
        print(f"   Top-{top_k} hit rate: {topk_hit_rate:.3f}")
        print(f"   Mean similarity (relevant):   {rel_mean:.4f}")
        print(f"   Mean similarity (irrelevant): {irrel_mean:.4f}")


def compare_models(
    query: str,
    index_name: str = DEFAULT_INDEX_NAME,
    api_keys_path: str = DEFAULT_API_KEYS_PATH,
    show_details: bool = True,
):
    """
    Compare all embedding models using Pinecone index retrieval.
    
    WARNING: For accurate comparison, the index must be created with the same
    embedding model. Using different models will give incorrect/incomparable results.
    
    Note: File type filtering is no longer supported. Use strategy parameter in retriever instead.
    """
    print("="*60)
    print("EMBEDDING MODEL COMPARISON (via Pinecone)")
    print("="*60)
    print(f"\nQuery: {query}")
    print(f"Index: {index_name}")
    print("\n⚠️  WARNING: Results only accurate if index was created with each model!")
    print("    For direct embedding comparison, use --similarity_test mode.\n")
    print(f"Testing {len(EMBEDDING_MODELS)} models...")
    
    results = []
    
    for model_info in EMBEDDING_MODELS:
        model_name = model_info["name"]
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        if model_info.get("recommended"):
            print("⭐ RECOMMENDED")
        print(f"Description: {model_info['description']}")
        
        result = test_model_with_index(
            model_name=model_name,
            query=query,
            index_name=index_name,
            api_keys_path=api_keys_path,
            top_k=5,
        )
        
        results.append(result)
        print_model_results(result, show_details=show_details)
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    # Filter successful results
    successful = [r for r in results if r["success"]]
    
    if not successful:
        print("\n❌ No models succeeded. Check errors above.")
        return
    
    # Sort by average score (higher is better)
    successful.sort(key=lambda x: x["avg_score"], reverse=True)
    
    print(f"\nRanked by average score (higher is better):\n")
    for i, result in enumerate(successful, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{medal} {result['model']}")
        print(f"   Avg score: {result['avg_score']:.4f} | "
              f"Max: {result['max_score']:.4f} | "
              f"Results: {result['num_results']}")
    
    # Best model
    best = successful[0]
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best['model']}")
    print(f"{'='*60}")
    print(f"Average score: {best['avg_score']:.4f}")
    print(f"Document types: {best['doc_types']}")
    
    if show_details:
        print(f"\nTop results from best model:")
        for i, result in enumerate(best['results'][:3], 1):
            print(f"\n  {i}. Score: {result['score']:.4f}")
            print(f"     Type: {result.get('doc_type', 'unknown')}")
            print(f"     Title: {result.get('title', 'N/A')}")
            print(f"     URL: {result.get('url', 'N/A')[:80]}...")
            content = result.get('chunk_text_only', result.get('text', ''))[:150]
            print(f"     Content: {content}...")


def main():
    """Main function with example queries."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare embedding models for Hebrew retrieval"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="איך משלמים ארנונה?",
        help="Query to test (default: Hebrew property tax question)"
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default=DEFAULT_INDEX_NAME,
        help="Pinecone index name"
    )
    parser.add_argument(
        "--api_keys_path",
        type=str,
        default=DEFAULT_API_KEYS_PATH,
        help="Path to API keys file"
    )
    parser.add_argument(
        "--no_details",
        action="store_true",
        help="Don't show detailed results for each model"
    )
    parser.add_argument(
        "--similarity_test",
        action="store_true",
        help="Test embedding similarity directly (no Pinecone index needed)"
    )
    parser.add_argument(
        "--eval_testset",
        type=str,
        default=None,
        help="Path to JSON/YAML test set for labeled evaluation"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top-k for hit rate when using --eval_testset (default: 3)"
    )
    
    args = parser.parse_args()
    
    if args.eval_testset:
        # Labeled evaluation on a test set
        try:
            test_queries = load_testset(args.eval_testset)
        except Exception as e:
            print(f"❌ Failed to load test set: {e}")
            sys.exit(1)

        evaluate_models_on_testset(
            test_queries=test_queries,
            top_k=args.top_k,
        )

    elif args.similarity_test:
        # Direct similarity comparison (single query, default docs)
        compare_similarity_direct(query=args.query)

    else:
        # Retrieval comparison (requires matching index)
        if not PINECONE_AVAILABLE:
            print("⚠️  ERROR: Pinecone dependencies not available.")
            print("    Use --similarity_test or --eval_testset for direct embedding comparison.\n")
            sys.exit(1)
        
        print("⚠️  NOTE: For accurate comparison, index must match the model.")
        print("    Use --similarity_test or --eval_testset for direct embedding comparison (no index needed).\n")
        compare_models(
            query=args.query,
            index_name=args.index_name,
            api_keys_path=args.api_keys_path,
            show_details=not args.no_details,
        )


if __name__ == "__main__":
    main()

