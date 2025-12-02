"""
Quick test to verify Gemini API call works correctly.

This script tests the basic Gemini integration to ensure
the API calls are formatted correctly.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from gemini_integration import init_gemini, call_gemini, load_api_keys


def test_gemini_basic():
    """Test basic Gemini API call."""
    print("=" * 60)
    print("TESTING GEMINI API CALL")
    print("=" * 60)
    
    # Load API keys
    try:
        api_keys = load_api_keys("../utils/api_keys.json")
    except FileNotFoundError:
        print("[ERROR] api_keys.json not found")
        print("[INFO] Please create utils/api_keys.json with GEMINI_API_KEY")
        return
    
    if "GEMINI_API_KEY" not in api_keys:
        print("[ERROR] GEMINI_API_KEY not found in utils/api_keys.json")
        return
    
    # Initialize Gemini
    print("\n[STEP] Initializing Gemini...")
    try:
        model = init_gemini(api_keys, "gemini-2.5-flash")
        print("[OK] Gemini initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Gemini: {e}")
        return
    
    # Test simple prompt
    print("\n[STEP] Testing simple prompt...")
    test_prompt = """אתה עוזר AI של עיריית חיפה.
    
שאלה: מה מספר הטלפון של המוקד העירוני?

ענה בקצרה."""
    
    try:
        response = call_gemini(model, test_prompt)
        print("[OK] Gemini API call successful")
        print("\n[RESPONSE]")
        print("-" * 60)
        print(response)
        print("-" * 60)
    except Exception as e:
        print(f"[ERROR] Failed to call Gemini: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_gemini_basic()

