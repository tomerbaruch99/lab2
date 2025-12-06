"""
Example usage of Smart Page Finder.

This demonstrates how to use the Smart Page Finder to get relevant page
recommendations based on user queries.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly to avoid circular dependencies
import importlib.util
smart_page_finder_path = project_root / "utils" / "smart_page_finder.py"
spec = importlib.util.spec_from_file_location("smart_page_finder", smart_page_finder_path)
smart_page_finder_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smart_page_finder_module)
SmartPageFinder = smart_page_finder_module.SmartPageFinder


def main():
    """Example usage of Smart Page Finder."""
    
    # Initialize the finder
    print("Initializing Smart Page Finder...")
    finder = SmartPageFinder()
    
    # Example queries
    queries = [
        "איך אני מחדש תו חניה?",
        "תשלום ארנונה",
        "רישום לגן ילדים",
        "תשלומי חניה"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # Find relevant pages
        results = finder.find_relevant_pages(query, top_k=3)
        
        # Format and print results
        print(finder.format_results(results, include_scores=False))
        print()


if __name__ == "__main__":
    main()

