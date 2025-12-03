"""
Helper script to delete and recreate a Pinecone index with the correct dimension
for the current embedding model.

This is useful when changing embedding models that have different dimensions.
"""

import sys
import os

# Add project root to path to import modules
# Since this file is now in utils/, we only need to go up one level
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinecone import Pinecone
from utils import (
    load_pinecone_api_key,
    create_index,
    DEFAULT_INDEX_NAME,
    DEFAULT_API_KEYS_PATH,
    DEFAULT_EMBEDDING_MODEL,
)
from utils.embedding import EmbeddingModel


def recreate_index(
    index_name: str,
    api_keys_path: str = DEFAULT_API_KEYS_PATH,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    confirm: bool = False,
):
    """
    Delete and recreate a Pinecone index with the correct dimension for the embedding model.
    
    Args:
        index_name: Name of the index to recreate
        api_keys_path: Path to API keys file
        embedding_model_name: Name of the embedding model (to get dimension)
        confirm: If False, will ask for confirmation before deleting
    """
    print("="*60)
    print("RECREATE PINECONE INDEX")
    print("="*60)
    print(f"\nIndex name: {index_name}")
    print(f"Embedding model: {embedding_model_name}")
    
    # Load embedding model to get dimension
    print(f"\n[STEP] Loading embedding model to get dimension...")
    embed_model = EmbeddingModel(embedding_model_name, verbose=True)
    required_dimension = embed_model.dimension
    print(f"[OK] Required dimension: {required_dimension}")
    
    # Initialize Pinecone
    print(f"\n[STEP] Initializing Pinecone...")
    pinecone_api_key = load_pinecone_api_key(api_keys_path)
    pc = Pinecone(api_key=pinecone_api_key)
    print("[OK] Pinecone initialized")
    
    # Check if index exists
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"\n[INFO] Index '{index_name}' does not exist. Creating new index...")
        create_index(
            pc,
            index_name,
            required_dimension,
            metric="cosine",
            cloud="aws",
            region="us-east-1",
        )
        print(f"\n[OK] Index '{index_name}' created successfully!")
        return
    
    # Get existing index info
    index_stats = pc.describe_index(index_name)
    existing_dimension = index_stats.dimension
    
    print(f"\n[INFO] Existing index dimension: {existing_dimension}")
    print(f"[INFO] Required dimension: {required_dimension}")
    
    if existing_dimension == required_dimension:
        print(f"\n[OK] Index dimension matches! No need to recreate.")
        return
    
    # Dimension mismatch - need to recreate
    print(f"\n[WARN] Dimension mismatch detected!")
    print(f"[WARN] The existing index has dimension {existing_dimension}")
    print(f"[WARN] but the embedding model requires dimension {required_dimension}")
    
    if not confirm:
        response = input(f"\n[CONFIRM] Delete and recreate index '{index_name}'? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("[CANCEL] Operation cancelled.")
            return
    
    print(f"\n[STEP] Deleting index '{index_name}'...")
    pc.delete_index(index_name)
    print(f"[OK] Index deleted")
    
    print(f"\n[STEP] Waiting for deletion to complete...")
    import time
    while index_name in [idx["name"] for idx in pc.list_indexes()]:
        time.sleep(1)
    print(f"[OK] Deletion confirmed")
    
    print(f"\n[STEP] Creating new index with correct dimension...")
    create_index(
        pc,
        index_name,
        required_dimension,
        metric="cosine",
        cloud="aws",
        region="us-east-1",
    )
    print(f"\n[OK] Index '{index_name}' recreated successfully!")
    print(f"[INFO] You can now run indexing.py to populate it with data.")


def main():
    """Main function with command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Recreate Pinecone index with correct dimension for embedding model"
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default=DEFAULT_INDEX_NAME,
        help="Name of the index to recreate"
    )
    parser.add_argument(
        "--api_keys_path",
        type=str,
        default=DEFAULT_API_KEYS_PATH,
        help="Path to API keys file"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name (to determine dimension)"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    recreate_index(
        index_name=args.index_name,
        api_keys_path=args.api_keys_path,
        embedding_model_name=args.embedding_model,
        confirm=args.yes,
    )


if __name__ == "__main__":
    main()

