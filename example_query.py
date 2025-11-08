from rag_retriever import retrieve
from what  import map_query_to_category, get_category_candidates

print("=" * 60)
print("Contract Clause Q&A Assistant")
print("=" * 60)
print("\nExample questions:")
print("  → 'Can the contract be terminated without cause?'")
print("  → 'What are the non-compete restrictions?'")
print("  → 'Is there a governing law clause?'")
print("\nFiltering options:")
print("  - Filter by filename: Enter 'y' when prompted, then provide the filename")
print("  - Category: Automatically detected from your question")
print("  - Press Enter to skip any filter")
print("\nType 'quit' or 'exit' to stop.\n")

while True:
    q = input("\nYour question: ").strip()
    if q.lower() in ("quit", "exit", "q"):
        break
    
    if not q:
        continue
    
    # Show query understanding
    category, confidence = map_query_to_category(q)
    if category:
        print(f"\n[Query Understanding] Detected category: {category} (confidence: {confidence:.2f})")
    
    # Ask for optional filters
    filename_filter = None
    category_filter = None
    
    # Ask about filename filter
    filename_choice_raw = input("\nFilter by specific filename? (y/n/list, or press Enter to skip): ").strip()
    filename_choice = filename_choice_raw.lower()
    
    if filename_choice in ('y', 'yes'):
        # Show some available filenames
        try:
            import pandas as pd
            df = pd.read_parquet("cuad_prepared_data/cuad_long_clauses.parquet")
            available = df["filename"].unique()[:10]
            print(f"\n  Sample filenames (showing first 10 of {df['filename'].nunique()}):")
            for f in available:
                print(f"    - {f}")
        except Exception:
            pass
        filename_input = input("\nEnter filename (partial match OK, or press Enter to skip): ").strip()
        if filename_input:
            filename_filter = filename_input
            print(f"  → Filtering by filename: {filename_filter}")
    elif filename_choice == 'list':
        # Show all available filenames
        try:
            import pandas as pd
            df = pd.read_parquet("cuad_prepared_data/cuad_long_clauses.parquet")
            available = sorted(df["filename"].unique())
            print(f"\n  Available filenames ({len(available)} total):")
            for f in available:
                print(f"    - {f}")
        except Exception as e:
            print(f"  Could not load filenames: {e}")
    elif filename_choice_raw:
        # If they entered something that's not y/yes/list, treat it as a filename
        # Preserve original case for filename matching
        filename_filter = filename_choice_raw
        print(f"  → Filtering by filename: {filename_filter}")
    
    # Category filter
    if category:
        # Auto-use detected category
        category_filter = category
        print(f"  → Using detected category: {category}")
    else:
        # Ask if they want to filter by category
        cat_choice = input("Filter by specific category? (y/n, or press Enter to skip): ").strip().lower()
        if cat_choice in ('y', 'yes'):
            cat_input = input("Enter category name: ").strip()
            if cat_input:
                category_filter = cat_input
                print(f"  → Filtering by category: {category_filter}")
    
    # Retrieve results
    print("\n[Retrieving...]")
    results = retrieve(
        q, 
        k_dense=8, 
        k_bm25=8, 
        final_k=5,
        filename=filename_filter,
        category=category_filter,
        use_query_understanding=True
    )
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Top {len(results)} Results:")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Category: {result.get('category', 'N/A')}")
        print(f"Filename: {result.get('filename', 'N/A')}")
        if result.get('answer'):
            print(f"Answer: {result['answer']}")
        print(f"\nText:\n{result['text']}\n")
        print("-" * 60)
