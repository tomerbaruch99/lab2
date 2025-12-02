"""
Example usage of Gemini RAG integration

This demonstrates the complete RAG pipeline:
1. Retrieve chunks from Pinecone
2. Build prompt with prompt_builder
3. Generate answer with Gemini
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_integration import GeminiRAG


def example_basic_rag():
    """Basic RAG example."""
    print("=" * 60)
    print("EXAMPLE 1: Basic RAG")
    print("=" * 60)
    
    # Initialize RAG system
    rag = GeminiRAG(
        api_keys_path="../utils/api_keys.json",
        gemini_model_name="gemini-2.5-flash",
    )
    
    # Ask a question
    question = "איך משלמים ארנונה?"
    result = rag.answer_question(question, top_k=5)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{result['answer']}")


def example_exclude_pdfs():
    """Example excluding PDFs to get HTML/TXT webpages."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Excluding PDFs")
    print("=" * 60)
    
    rag = GeminiRAG(api_keys_path="../utils/api_keys.json")
    
    question = "מה מספר הטלפון של המוקד העירוני?"
    result = rag.answer_question(
        question,
        top_k=5,
        exclude_file_types=["pdf"],  # Only HTML/TXT webpages
    )
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer (from HTML/TXT only):\n{result['answer']}")


def example_with_chunks():
    """Example showing retrieved chunks."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: With Retrieved Chunks")
    print("=" * 60)
    
    rag = GeminiRAG(api_keys_path="../utils/api_keys.json")
    
    question = "איך מזמינים תור?"
    result = rag.answer_question(
        question,
        top_k=3,
        return_chunks=True,
    )
    
    print(f"\nQuestion: {question}")
    print(f"\nRetrieved {len(result['chunks'])} chunks:")
    for i, chunk in enumerate(result['chunks'], 1):
        print(f"\n  Chunk {i}:")
        print(f"    Title: {chunk.get('title', 'N/A')}")
        print(f"    URL: {chunk.get('url', 'N/A')}")
        print(f"    Score: {chunk['score']:.4f}")
    
    print(f"\nAnswer:\n{result['answer']}")


def example_conversation():
    """Example with conversation history."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Conversational RAG")
    print("=" * 60)
    
    rag = GeminiRAG(api_keys_path="../utils/api_keys.json")
    
    # First turn
    history = []
    question1 = "מה שעות הפעילות של המוקד העירוני?"
    result1 = rag.answer_question(question1, top_k=3)
    
    history.append({"role": "user", "content": question1})
    history.append({"role": "assistant", "content": result1["answer"]})
    
    print(f"User: {question1}")
    print(f"Assistant: {result1['answer']}\n")
    
    # Second turn (with history)
    question2 = "ואיך משלמים ארנונה?"
    result2 = rag.answer_with_conversation(question2, history, top_k=3)
    
    print(f"User: {question2}")
    print(f"Assistant: {result2['answer']}")


def example_custom_instruction():
    """Example with custom system instruction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Custom Instruction")
    print("=" * 60)
    
    rag = GeminiRAG(api_keys_path="../utils/api_keys.json")
    
    custom_instruction = """אתה עוזר AI של עיריית חיפה.
ענה בקצרה ובבהירות.
השתמש רק במידע מהמסמכים."""
    
    question = "מה מספר הטלפון של המוקד?"
    result = rag.answer_question(
        question,
        top_k=3,
        custom_instruction=custom_instruction,
    )
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer (with custom instruction):\n{result['answer']}")


if __name__ == "__main__":
    # Uncomment examples to run
    
    # example_basic_rag()
    # example_exclude_pdfs()
    # example_with_chunks()
    # example_conversation()
    # example_custom_instruction()
    
    print("Uncomment examples in the code to run them.")

