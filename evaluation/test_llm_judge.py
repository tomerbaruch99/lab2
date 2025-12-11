"""
Quick test script to verify LLM judge is working correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gemini_integration import init_gemini, load_api_keys
from evaluation.llm_judge import judge_answer
from utils import DEFAULT_API_KEYS_PATH, DEFAULT_LLM_JUDGE_MODEL

def test_judge():
    """Test the LLM judge with a simple example."""
    print("Testing LLM Judge...")
    print(f"Using model: {DEFAULT_LLM_JUDGE_MODEL}\n")
    
    # Load API keys
    api_keys = load_api_keys(DEFAULT_API_KEYS_PATH)
    
    # Initialize model
    try:
        gemini_model = init_gemini(api_keys, DEFAULT_LLM_JUDGE_MODEL)
        print(f"[OK] Model initialized: {DEFAULT_LLM_JUDGE_MODEL}\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}")
        print(f"[INFO] Trying alternative model: gemini-2.0-flash")
        try:
            gemini_model = init_gemini(api_keys, "gemini-2.0-flash")
            print(f"[OK] Model initialized: gemini-2.0-flash\n")
        except Exception as e2:
            print(f"[ERROR] Failed with alternative model too: {e2}")
            return
    
    # Test with a simple example
    question = "איך משלמים ארנונה?"
    gold_answer = "ניתן לשלם ארנונה באתר האינטרנט, במוקד הטלפוני, או בקופת העירייה."
    rag_answer = "ניתן לשלם ארנונה באתר האינטרנט או במוקד הטלפוני."
    
    print("Test case:")
    print(f"  Question: {question}")
    print(f"  Gold answer: {gold_answer}")
    print(f"  RAG answer: {rag_answer}\n")
    
    print("Calling judge...")
    try:
        scores = judge_answer(
            question=question,
            gold_answer=gold_answer,
            rag_answer=rag_answer,
            gemini_model=gemini_model,
        )
        
        print("\n[OK] Judge returned scores:")
        for key, value in scores.items():
            print(f"  {key}: {value}")
        
        # Check if all zeros
        if all(v == 0.0 for v in scores.values()):
            print("\n[WARN] All scores are 0.0 - this might indicate a parsing issue!")
        else:
            print("\n[OK] Judge is working correctly!")
            
    except Exception as e:
        print(f"\n[ERROR] Judge failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_judge()

