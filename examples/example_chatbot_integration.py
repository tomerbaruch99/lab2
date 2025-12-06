"""
Example: Integrating Smart Page Finder with Chatbot

This shows how to integrate the Smart Page Finder into the chatbot application
to suggest relevant official pages alongside RAG responses.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Smart Page Finder (using direct import to avoid circular dependencies)
import importlib.util
smart_page_finder_path = project_root / "utils" / "smart_page_finder.py"
spec = importlib.util.spec_from_file_location("smart_page_finder", smart_page_finder_path)
smart_page_finder_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smart_page_finder_module)
SmartPageFinder = smart_page_finder_module.SmartPageFinder


def get_chatbot_response_with_page_suggestions(query: str, rag_response: str):
    """
    Example function showing how to integrate Smart Page Finder with chatbot.
    
    Args:
        query: User's query
        rag_response: The RAG-generated response
    
    Returns:
        Enhanced response with page suggestions
    """
    # Initialize Smart Page Finder (reuse instance for efficiency)
    finder = SmartPageFinder()
    
    # Find relevant pages
    relevant_pages = finder.find_relevant_pages(query, top_k=3)
    
    # Format the response
    response = rag_response
    
    if relevant_pages:
        response += "\n\n---\n\n"
        response += "**למידע נוסף - דפים רלוונטיים באתר העירייה:**\n\n"
        
        for i, page in enumerate(relevant_pages, 1):
            title = page['title']
            subtitle = page.get('subtitle', '')
            url = page['url']
            
            # Format title with subtitle if available
            if subtitle and subtitle != title:
                display_title = f"{title} - {subtitle}"
            else:
                display_title = title
            
            # Format as markdown link (matching chatbot format)
            response += f"{i}. [{display_title}]({url})\n"
    
    return response


def example_usage():
    """Example of how to use this in the chatbot."""
    
    # Simulate a user query
    user_query = "איך אני מחדש תו חניה?"
    
    # Simulate RAG response (in real chatbot, this comes from your RAG system)
    rag_response = "כדי לחדש תו חניה, יש לפנות למחלקת החניה בעירייה..."
    
    # Get enhanced response with page suggestions
    enhanced_response = get_chatbot_response_with_page_suggestions(user_query, rag_response)
    
    print("=" * 60)
    print("User Query:", user_query)
    print("=" * 60)
    print("\nEnhanced Response with Page Suggestions:")
    print("-" * 60)
    print(enhanced_response)
    print("=" * 60)


if __name__ == "__main__":
    example_usage()

