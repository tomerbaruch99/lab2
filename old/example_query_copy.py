from rag_retriever_copy import retrieve
from query_understanding_copy import map_query_to_category, get_category_candidates

import pandas as pd

print("=" * 60)
print("Contract Clause Q&A Assistant — Retrieval Demo")
print("=" * 60)
print("\nExample questions:")
print("  → 'Can the contract be terminated without cause?'")
print("  → 'What are the non-compete restrictions?'")
print("  → 'Is there a governing law clause?'")
print("\nFiltering options:")
print("  - Filter by filename: Enter 'y' when prompted, then provide the filename")
print("  - Category: Automatically detected from your question (override optional)")
print("  - Press Enter to skip any filter")
print("\nType 'quit' or 'exit' to stop.\n")

while True:
    q = input("\nYour question: ").strip()
    if q.lower() in ("quit", "exit", "q"):
        break
    if not q:
        continue

    # Query understanding preview
    cat, conf = map_query_to_category(q)
    if cat:
        print(f"\n[Query Understanding] Detected category: {cat} (confidence: {conf:.2f})")
        print("Top-3 candidates:")
        for c, s in get_category_candidates(q, top_k=3):
            print(f"  - {c} ({s:.2f})")

    # Optional filters
    filename_filter = None
    category_filter = None

    # Filename filter (with quick listing)
    try:
        df = pd.read_parquet("cuad_prepared_data/cuad_long_clauses.parquet")
    except Exception:
        df = None

    filename_choice_raw = input("\nFilter by specific filename? (y/n/list, or press Enter to skip): ").strip()
    filename_choice = filename_choice_raw.lower()

    if filename_choice in ('y', 'yes'):
        if df is not None:
            available = df["filename"].unique()[:10]
            print(f"\n  Sample filenames (showing first 10 of {df['filename'].nunique()}):")
            for f in available:
                print(f"    - {f}")
        filename_input = input("\nEnter filename (partial match OK, or press Enter to skip): ").strip()
        if filename_input:
            filename_filter = filename_input
            print(f"  → Filtering by filename: {filename_filter}")
    elif filename_choice == 'list':
        if df is not None:
            available = sorted(df["filename"].unique())
            print(f"\n  Available filenames ({len(available)} total):")
            for f in available:
                print(f"    - {f}")
        else:
            print("  Could not load filenames.")
    elif filename_choice_raw:
        filename_filter = filename_choice_raw
        print(f"  → Filtering by filename: {filename_filter}")

    # Category filter
    if cat:
        category_filter = cat
        print(f"  → Using detected category: {category_filter}")
    else:
        cat_choice = input("Filter by specific category? (y/n, or press Enter to skip): ").strip().lower()
        if cat_choice in ('y', 'yes'):
            cat_input = input("Enter category name: ").strip()
            if cat_input:
                category_filter = cat_input
                print(f"  → Filtering by category: {category_filter}")

    # Retrieve
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

    # Display
    print(f"\n{'=' * 60}")
    print(f"Top {len(results)} Results:")
    print(f"{'=' * 60}\n")

    for i, r in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Category: {r.get('category', 'N/A')}")
        print(f"Filename: {r.get('filename', 'N/A')}")
        if r.get('answer'):
            print(f"Answer: {r['answer']}")
        print(f"\nText:\n{r['text']}\n")
        print("-" * 60)
