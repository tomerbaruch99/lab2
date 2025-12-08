"""
Build page_index.csv from scraped Haifa municipality data.

This script:
1. Loads the scraped JSON data
2. Extracts title, subtitle, and URL for each page
3. Generates embeddings for page titles (or title + subtitle)
4. Saves to CSV: page_index.csv with (title, url, embedding)
"""

import json
import csv
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly from modules to avoid circular dependencies
import importlib.util
embedding_path = Path(__file__).parent / "embedding.py"
spec = importlib.util.spec_from_file_location("embedding", embedding_path)
embedding_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embedding_module)
EmbeddingModel = embedding_module.EmbeddingModel

config_path = Path(__file__).parent / "config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
DEFAULT_EMBEDDING_MODEL = config_module.DEFAULT_EMBEDDING_MODEL


def build_page_index(
    json_path: str = None,
    output_path: str = None,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
):
    """
    Build page index CSV from scraped JSON data.
    
    Args:
        json_path: Path to the scraped JSON file (defaults to scrape_and_prepare_data/haifa_scraped.json)
        output_path: Path to output CSV file (defaults to scrape_and_prepare_data/page_index.csv)
        embedding_model_name: Name of the embedding model to use
    """
    # Set default paths (relative to current working directory - main project directory)
    if json_path is None:
        json_path = "scrape_and_prepare_data/haifa_scraped.json"
    if output_path is None:
        output_path = "scrape_and_prepare_data/page_index.csv"
    
    json_path = Path(json_path)
    output_path = Path(output_path)
    
    # Load scraped data
    print(f"[STEP] Loading scraped data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        pages = json.load(f)
    
    print(f"[INFO] Found {len(pages)} pages")
    
    # Prepare page data
    page_data = []
    for page in pages:
        url = page.get('url', '')
        title = page.get('title', '').strip()
        subtitle = page.get('subtitle', '').strip()
        
        # Skip pages without title or URL
        if not title or not url:
            continue
        
        # Combine title and subtitle for better semantic matching
        # Use title + subtitle if subtitle exists and is meaningful
        if subtitle and subtitle != title:
            text_to_embed = f"{title} {subtitle}"
        else:
            text_to_embed = title
        
        page_data.append({
            'title': title,
            'subtitle': subtitle,
            'url': url,
            'text_to_embed': text_to_embed
        })
    
    print(f"[INFO] Processing {len(page_data)} pages with valid titles")
    
    # Initialize embedding model (force CPU to avoid CUDA compatibility issues)
    print(f"[STEP] Loading embedding model: {embedding_model_name}")
    embed_model = EmbeddingModel(embedding_model_name, device="cpu", verbose=True)
    
    # Extract texts to embed
    texts_to_embed = [p['text_to_embed'] for p in page_data]
    
    # Generate embeddings
    print(f"[STEP] Generating embeddings for {len(texts_to_embed)} pages...")
    embeddings = embed_model.embed(texts_to_embed, show_progress=True)
    
    # Write to CSV
    print(f"[STEP] Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['title', 'subtitle', 'url', 'embedding'])
        
        # Write data
        for page, embedding in zip(page_data, embeddings):
            # Convert embedding list to string representation
            embedding_str = ','.join(map(str, embedding))
            writer.writerow([
                page['title'],
                page['subtitle'],
                page['url'],
                embedding_str
            ])
    
    print(f"[DONE] Page index saved to {output_path}")
    print(f"[INFO] Total pages indexed: {len(page_data)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build page index CSV from scraped data")
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Path to scraped JSON file (defaults to scrape_and_prepare_data/haifa_scraped.json)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to output CSV file (defaults to scrape_and_prepare_data/page_index.csv)"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name"
    )
    
    args = parser.parse_args()
    
    build_page_index(
        json_path=args.json_path,
        output_path=args.output_path,
        embedding_model_name=args.embedding_model
    )

