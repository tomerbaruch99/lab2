"""
Example usage of the Prompt Builder

This demonstrates how to use the prompt builder with retrieved chunks.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever import Retriever
from prompt_builder import PromptBuilder, PromptStyle


def example_basic_prompt():
    """Basic prompt building example."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Prompt Building")
    print("=" * 60)
    
    # Initialize retriever and get chunks
    retriever = Retriever(
        api_keys_path="./utils/api_keys.json",
        embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        index_name="haifa-municipality-rag-index",
    )
    
    question = "איך משלמים ארנונה?"
    chunks = retriever.retrieve(question, top_k=3)
    
    # Build prompt
    builder = PromptBuilder()
    prompt = builder.build_prompt(question, chunks)
    
    print("\nQuestion:", question)
    print("\nRetrieved", len(chunks), "chunks")
    print("\n" + "=" * 60)
    print("GENERATED PROMPT:")
    print("=" * 60)
    print(prompt)


def example_different_styles():
    """Example showing different prompt styles."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Different Prompt Styles")
    print("=" * 60)
    
    # Sample chunks
    sample_chunks = [
        {
            "id": "doc1::chunk-0",
            "score": 0.85,
            "text": "כותרת: מוקדי השירות\n\nמוקד עירוני 106\nטלפון: 04-8356356",
            "chunk_text_only": "מוקד עירוני 106\nטלפון: 04-8356356",
            "url": "https://www.haifa.muni.il/resident-service/service-center/",
            "title": "מוקדי השירות",
        },
    ]
    
    question = "מה מספר הטלפון של המוקד העירוני?"
    
    styles = [
        PromptStyle.DETAILED,
        PromptStyle.CONCISE,
        PromptStyle.CONVERSATIONAL,
        PromptStyle.STRUCTURED,
    ]
    
    for style in styles:
        print(f"\n--- {style.value.upper()} STYLE ---")
        builder = PromptBuilder(style=style)
        prompt = builder.build_prompt(question, sample_chunks)
        print(prompt[:300] + "..." if len(prompt) > 300 else prompt)


def example_conversational_chat():
    """Example with conversation history."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Conversational Chat with History")
    print("=" * 60)
    
    builder = PromptBuilder()
    
    # Conversation history
    history = [
        {"role": "user", "content": "מה שעות הפעילות של המוקד?"},
        {"role": "assistant", "content": "המוקד העירוני 106 פעיל 24 שעות ביממה."},
        {"role": "user", "content": "ואיך משלמים ארנונה?"},
    ]
    
    # Retrieve chunks for current question
    retriever = Retriever(
        api_keys_path="./utils/api_keys.json",
        index_name="haifa-municipality-rag-index",
    )
    chunks = retriever.retrieve("איך משלמים ארנונה?", top_k=2)
    
    # Build prompt with conversation context
    # Note: PromptBuilder doesn't have build_chat_prompt, so we use build_prompt
    # and include conversation history in custom_instruction if needed
    current_question = history[-1]["content"]
    prompt = builder.build_prompt(current_question, chunks)
    
    print(prompt)


def example_custom_instruction():
    """Example with custom system instruction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Custom System Instruction")
    print("=" * 60)
    
    custom_instruction = """אתה עוזר AI של עיריית חיפה.
ענה בקצרה ובבהירות.
השתמש רק במידע מהמסמכים."""
    
    builder = PromptBuilder(system_instruction=custom_instruction)
    
    sample_chunks = [
        {
            "text": "תשלום ארנונה ניתן לבצע דרך האתר או במוקד.",
            "url": "https://www.haifa.muni.il/arnona/",
        },
    ]
    
    prompt = builder.build_prompt("איך משלמים ארנונה?", sample_chunks)
    print(prompt)


def example_without_sources():
    """Example without source citations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Prompt Without Sources")
    print("=" * 60)
    
    builder = PromptBuilder()
    
    sample_chunks = [
        {
            "text": "מוקד עירוני 106\nטלפון: 04-8356356",
            "url": "https://www.haifa.muni.il/service-center/",
            "title": "מוקדי השירות",
        },
    ]
    
    # Build without sources
    prompt = builder.build_prompt(
        "מה מספר הטלפון של המוקד?",
        sample_chunks,
        include_sources=False
    )
    
    print(prompt)


if __name__ == "__main__":
    # Uncomment examples to run
    
    # example_basic_prompt()
    example_different_styles()
    # example_conversational_chat()
    # example_custom_instruction()
    # example_without_sources()

