"""
Pinecone utility functions shared across the RAG system.
"""

import os
import json
from typing import Optional

from pinecone import Pinecone, ServerlessSpec


def load_pinecone_api_key(api_keys_path: str) -> str:
    """Load Pinecone API key from env or from api_keys.json."""
    env_key = os.getenv("PINECONE_API_KEY")
    if env_key:
        return env_key

    if not os.path.exists(api_keys_path):
        raise FileNotFoundError(
            f"api_keys.json not found at {api_keys_path} and PINECONE_API_KEY env var not set."
        )

    with open(api_keys_path, "r", encoding="utf-8") as f:
        api_keys = json.load(f)

    if "PINECONE_API_KEY" not in api_keys:
        raise KeyError("Key 'PINECONE_API_KEY' missing from api_keys.json")

    return api_keys["PINECONE_API_KEY"]


def create_index(
    pc: Pinecone,
    index_name: str,
    dimension: int,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
    """
    Create Pinecone index if it doesn't exist.
    Checks dimension mismatch and raises error if existing index has wrong dimension.
    
    Args:
        pc: Pinecone client instance
        index_name: Name of the index to create
        dimension: Dimension of the vectors
        metric: Distance metric (default: "cosine")
        cloud: Cloud provider (default: "aws")
        region: AWS region (default: "us-east-1")
        
    Raises:
        ValueError: If index exists but has wrong dimension
    """
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"[INFO] Creating index '{index_name}' (dim={dimension}, metric='{metric}')...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"[OK] Index '{index_name}' created successfully")
    else:
        # Check if existing index has the correct dimension
        index_stats = pc.describe_index(index_name)
        existing_dimension = index_stats.dimension
        
        if existing_dimension != dimension:
            raise ValueError(
                f"Index '{index_name}' already exists but has dimension {existing_dimension}, "
                f"while the embedding model requires dimension {dimension}. "
                f"You need to either:\n"
                f"  1. Delete the existing index and recreate it: "
                f"     pc.delete_index('{index_name}')\n"
                f"  2. Use a different index name\n"
                f"  3. Use the embedding model that matches the existing dimension"
            )
        print(f"[INFO] Index '{index_name}' already exists (dim={dimension}, metric='{metric}')")

