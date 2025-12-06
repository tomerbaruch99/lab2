"""
Check if Pinecone index contains chunks with non-empty links metadata.

Usage:
    python check_pinecone_links.py \
        --api_keys_path utils/api_keys.json \
        --index_name haifa-municipality-rag-index \
        --embedding_model paraphrase-multilingual-MiniLM-L12-v2 \
        --sample_size 100
"""

import argparse
import json
from typing import List

from pinecone import Pinecone
from utils import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_NAME,
    DEFAULT_API_KEYS_PATH,
    load_pinecone_api_key,
    EmbeddingModel,
)


def check_links_in_index(
    api_keys_path: str,
    index_name: str,
    embedding_model_name: str,
    sample_size: int = 100,
) -> None:
    """
    Check if any chunks in Pinecone have non-empty links metadata.
    
    Args:
        api_keys_path: Path to API keys JSON file
        index_name: Name of Pinecone index
        embedding_model_name: Name of embedding model (to get vector dimension)
        sample_size: Number of vectors to sample per namespace
    """
    # 1. Initialize Pinecone
    print(f"[STEP] Connecting to Pinecone...")
    api_key = load_pinecone_api_key(api_keys_path)
    pc = Pinecone(api_key=api_key)
    
    # 2. Get index
    try:
        index = pc.Index(index_name)
        print(f"[OK] Connected to index '{index_name}'")
    except Exception as e:
        print(f"[ERROR] Failed to connect to index: {e}")
        return
    
    # 3. Get index stats to see available namespaces
    print(f"[STEP] Getting index statistics...")
    namespace_counts = {}
    try:
        stats = index.describe_index_stats()
        print(f"[OK] Index stats retrieved")
        
        # Get namespace counts - handle both dict and object-style access
        if hasattr(stats, 'namespaces'):
            namespace_counts = stats.namespaces or {}
        elif isinstance(stats, dict):
            namespace_counts = stats.get('namespaces', {})
        else:
            # Try to convert to dict
            try:
                namespace_counts = dict(stats.namespaces) if hasattr(stats, 'namespaces') else {}
            except:
                pass
        
        if namespace_counts:
            print(f"\nNamespaces found:")
            for ns, ns_stats in namespace_counts.items():
                # Handle both dict and object-style stats
                if isinstance(ns_stats, dict):
                    count = ns_stats.get('vector_count', 0)
                elif hasattr(ns_stats, 'vector_count'):
                    count = ns_stats.vector_count
                else:
                    count = ns_stats if isinstance(ns_stats, (int, float)) else 0
                print(f"  - {ns}: {count:,} vectors")
        else:
            print("[INFO] No namespace-specific stats available")
            
    except Exception as e:
        print(f"[WARN] Could not get index stats: {e}")
        namespace_counts = {}
    
    # 4. Get vector dimension
    print(f"\n[STEP] Getting embedding model dimension...")
    embed_model = EmbeddingModel(embedding_model_name, device="cpu", verbose=False)
    dimension = embed_model.dimension
    print(f"[OK] Vector dimension: {dimension}")
    
    # 5. Prepare a dummy query vector (zeros) to query across namespaces
    dummy_vector = [0.0] * dimension
    
    # 6. Known namespaces (from data_preparation.py)
    known_namespaces = [
        "arnona", "water", "education", "sanitation", "parking",
        "emergency", "engineering", "welfare", "business", "culture", "general"
    ]
    
    # Use namespaces from stats if available, otherwise use known namespaces
    namespaces_to_check = list(namespace_counts.keys()) if namespace_counts else known_namespaces
    
    if not namespaces_to_check:
        # If no namespace info, try querying without namespace (default)
        namespaces_to_check = [None]
    
    print(f"\n[STEP] Querying {len(namespaces_to_check)} namespace(s) to check for links...")
    
    # 7. Query each namespace and check for links
    found_links = False
    chunks_with_links = 0
    total_chunks_checked = 0
    examples = []
    empty_link_examples = []  # Store examples of empty links for debugging
    
    for namespace in namespaces_to_check:
        ns_label = namespace if namespace else "(default)"
        print(f"\n  Checking namespace: {ns_label}...")
        
        try:
            # Query with dummy vector to get sample chunks
            results = index.query(
                vector=dummy_vector,
                top_k=min(sample_size, 100),  # Pinecone max is usually 10000, but 100 is enough for sampling
                include_metadata=True,
                namespace=namespace if namespace else None,
            )
            
            namespace_checked = len(results.matches)
            total_chunks_checked += namespace_checked
            print(f"    Retrieved {namespace_checked} chunks")
            
            # Check each match for links
            for match in results.matches:
                metadata = match.metadata or {}
                links_field = metadata.get("links", "")
                
                # Check if links is non-empty
                is_non_empty = False
                
                if links_field:
                    # links_field could be a JSON string or already a list
                    if isinstance(links_field, str):
                        links_str = links_field.strip()
                        # Check if it's not empty and not just "[]"
                        if links_str and links_str != "[]" and links_str.lower() not in ["null", "none", ""]:
                            try:
                                # Try to parse as JSON
                                parsed = json.loads(links_str)
                                if isinstance(parsed, list) and len(parsed) > 0:
                                    is_non_empty = True
                            except (json.JSONDecodeError, ValueError):
                                # If it's not valid JSON but not empty, might be a URL string
                                if len(links_str) > 4:  # At least "http"
                                    is_non_empty = True
                    elif isinstance(links_field, list):
                        if len(links_field) > 0:
                            is_non_empty = True
                    elif isinstance(links_field, dict):
                        # Could be a link object
                        if links_field.get("url"):
                            is_non_empty = True
                
                if is_non_empty:
                    chunks_with_links += 1
                    found_links = True
                    
                    # Store first few examples
                    if len(examples) < 5:
                        examples.append({
                            "namespace": ns_label,
                            "id": match.id,
                            "links": links_field,
                            "url": metadata.get("url", ""),
                            "title": metadata.get("title", "")[:50],
                        })
                else:
                    # Store examples of empty links for debugging (first 3)
                    if len(empty_link_examples) < 3:
                        empty_link_examples.append({
                            "namespace": ns_label,
                            "id": match.id,
                            "links_raw": links_field,
                            "links_type": type(links_field).__name__,
                            "links_str": str(links_field)[:100] if links_field is not None else "None",
                            "url": metadata.get("url", "")[:50],
                        })
            
            # Count chunks with links in this namespace (for reporting)
            chunks_with_links_in_ns = 0
            for match in results.matches:
                metadata = match.metadata or {}
                links_field = metadata.get("links", "")
                if links_field and str(links_field).strip() not in ["[]", "", "null", "None"]:
                    try:
                        if isinstance(links_field, str):
                            parsed = json.loads(links_field)
                            if isinstance(parsed, list) and len(parsed) > 0:
                                chunks_with_links_in_ns += 1
                        elif isinstance(links_field, (list, dict)) and len(links_field) > 0:
                            chunks_with_links_in_ns += 1
                    except:
                        if len(str(links_field).strip()) > 4:
                            chunks_with_links_in_ns += 1
            
            if namespace_checked > 0:
                print(f"    Found {chunks_with_links_in_ns} chunks with links in this namespace")
                
        except Exception as e:
            print(f"    [WARN] Error querying namespace {ns_label}: {e}")
            continue
    
    # 8. Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total chunks sampled: {total_chunks_checked:,}")
    print(f"Chunks with non-empty links: {chunks_with_links:,}")
    print(f"\nResult: {'✓ FOUND chunks with links!' if found_links else '✗ NO chunks with non-empty links found'}")
    
    if examples:
        print(f"\nExamples of chunks with links:")
        for i, ex in enumerate(examples, 1):
            print(f"\n  {i}. Namespace: {ex['namespace']}")
            print(f"     ID: {ex['id']}")
            print(f"     URL: {ex['url']}")
            print(f"     Title: {ex['title']}")
            links_preview = str(ex['links'])[:100] + "..." if len(str(ex['links'])) > 100 else str(ex['links'])
            print(f"     Links: {links_preview}")
    
    if not found_links and empty_link_examples:
        print(f"\nDEBUG: Examples of chunks with empty/missing links:")
        for i, ex in enumerate(empty_link_examples, 1):
            print(f"\n  {i}. Namespace: {ex['namespace']}")
            print(f"     ID: {ex['id']}")
            print(f"     URL: {ex['url']}")
            print(f"     Links field type: {ex['links_type']}")
            print(f"     Links value: {repr(ex['links_raw'])}")
            print(f"     Links as string: {ex['links_str']}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Check if Pinecone index contains chunks with non-empty links"
    )
    parser.add_argument(
        "--api_keys_path",
        type=str,
        default=DEFAULT_API_KEYS_PATH,
        help="Path to API keys JSON file"
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default=DEFAULT_INDEX_NAME,
        help="Name of Pinecone index"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Name of embedding model (to get vector dimension)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of vectors to sample per namespace (default: 100)"
    )
    
    args = parser.parse_args()
    
    check_links_in_index(
        api_keys_path=args.api_keys_path,
        index_name=args.index_name,
        embedding_model_name=args.embedding_model,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()

