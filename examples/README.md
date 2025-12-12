# Examples

Example scripts demonstrating how to use different components of the RAG system.

## Available Examples

- **`example_retriever_usage.py`** - Basic retrieval examples
  - Initialize retriever
  - Query Pinecone for relevant chunks
  - Filter by chunking strategy
  - Namespace detection

- **`example_prompt_builder.py`** - Prompt building examples
  - Different prompt styles (detailed, concise, conversational, structured)
  - Building prompts with retrieved chunks
  - Custom system instructions

- **`example_gemini_rag.py`** - Complete RAG pipeline
  - Full workflow (retrieval → prompt building → generation)
  - Using different chunking strategies
  - Conversation history support

- **`example_smart_page_finder.py`** - Smart Page Finder usage
  - Finding relevant pages based on user queries
  - Formatting page recommendations
  - Integration with chatbot responses

- **`example_retrieval_diagnostics.py`** - Retrieval diagnostics
  - Diagnose why retrieval might return poor results
  - File type distribution in results
  - Namespace detection accuracy

- **`example_chatbot_integration.py`** - Chatbot integration example
  - Complete integration example

- **`test_gemini_call.py`** - Simple Gemini API test
  - Verify Gemini API connection
  - Test API keys configuration

## Usage

Run any example directly:

```bash
python examples/example_retriever_usage.py
```

**Requirements:**
- API keys in `utils/api_keys.json`
- Data indexed in Pinecone (for retrieval examples)
- See comments in each example file for specific requirements

