"""
Haifa Municipality RAG Chatbot
================================

A Streamlit-based web interface for the Haifa Municipality RAG system.
This chatbot provides an interactive Hebrew (RTL) interface where users can:
- Ask questions about municipal services, regulations, and information
- Receive RAG-powered answers using Gemini 3 Pro (default: gemini-3-pro-preview)
- View answer confidence scores with visual indicators
- Get automatic page recommendations from the official Haifa website
- Maintain conversation history and export chat logs

The chatbot integrates three main components:
1. GeminiRAG: Complete RAG pipeline (retrieval → prompt building → generation)
2. SmartPageFinder: Semantic page recommendation system
3. Confidence Meter: Answer quality and reliability scoring

Features:
- Full Hebrew (RTL) support with Gisha font
- Real-time question answering
- Visual confidence indicators (🟢/🟡/🔴)
- Automatic relevant page suggestions
- Session-based chat history
- Export functionality for conversations

Usage:
    streamlit run chatbot.py

Requirements:
    - utils/api_keys.json with PINECONE_API_KEY and GEMINI_API_KEY
    - Data indexed in Pinecone (run indexing.py first)
    - scrape_and_prepare_data/page_index.csv (for Smart Page Finder)
"""

import warnings
import sys
from pathlib import Path
import streamlit as st

# Add project root to Python path to enable imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gemini_integration import GeminiRAG
from utils.smart_page_finder import SmartPageFinder
from utils import DEFAULT_API_KEYS_PATH, DEFAULT_INDEX_NAME



# -----------------------------
# PAGE CONFIG
# -----------------------------
# Configure Streamlit page settings for Hebrew RTL interface
st.set_page_config(
    page_title="צ'אטבוט עיריית חיפה",
    page_icon="logos/logo1.png",
    layout="centered"  # Centered layout for better readability
)


# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

html, body, [data-testid="stAppViewContainer"] {
    direction: rtl !important;
    text-align: right !important;
    background-color: #f3f9f5 !important;
    font-family: Gisha, Arial, sans-serif !important;
}

/* Hide ALL sidebar toggle buttons */
button[kind="header"],
button[title="Open sidebar"],
button[title="Close sidebar"],
[data-testid="baseButton-header"] {
    display: none !important;
}

/* ----- CENTER PAGE WIDTH ----- */
.block-container {
    max-width: 820px !important;
    margin: auto !important;
}

/* ----- TITLE ----- */
h1 {
    color: #195c8c !important;
    font-weight: bold !important;
    text-align: center !important;
}

/* ----- CHAT BUBBLES ----- */
.chat-bubble {
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    white-space: normal !important;
    line-height: 1.6 !important;
}

.user-bubble {
    background-color: #e6f3ff;
    border-right: 6px solid #4FA3D1;
}

.assistant-bubble {
    background-color: #ffffff;
    border-right: 6px solid #7CC242;
}

/* ----- INPUT FIELDS ----- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    direction: rtl !important;
    text-align: right !important;
}

/* ----- BUTTONS ----- */
button {
    background-color: #195c8c !important;
    color: white !important;
    border-radius: 8px !important;
    font-size: 16px !important;
}
button:hover {
    background-color: #4fa3d1 !important;
}

/* LINKS */
a {
    color: #195c8c !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)



# -----------------------------
# HEADER LOGO
# -----------------------------
st.markdown("""
<div style="text-align:center; margin-top:10px; margin-bottom:25px;">
""", unsafe_allow_html=True)

st.image("logos/logo2.png", width=260)

st.markdown("</div>", unsafe_allow_html=True)



# -----------------------------
# INITIALIZE SYSTEM
# -----------------------------
# Initialize session state for chat history
# Session state persists across reruns within the same session
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def init_rag():
    """
    Initialize RAG system and Smart Page Finder with caching.
    
    Uses Streamlit's @st.cache_resource decorator to cache the initialization
    across reruns, avoiding redundant API key loading and model initialization.
    
    Returns:
        Tuple of (GeminiRAG instance, SmartPageFinder instance)
        Returns (None, None) if initialization fails
    
    Note:
        This function is cached, so it only runs once per session.
        If initialization fails, the error is displayed to the user.
    """
    try:
        # Initialize RAG system with default API keys path and small chunks index
        rag_system = GeminiRAG(
            api_keys_path=DEFAULT_API_KEYS_PATH,
            index_name=DEFAULT_INDEX_NAME  # Using small chunks index
        )
        
        # Initialize Smart Page Finder for relevant page recommendations
        page_finder = SmartPageFinder()
        
        return rag_system, page_finder
    except Exception as e:
        # Display error to user if initialization fails
        st.error(f"שגיאה באתחול המערכת: {e}")
        return None, None

# Initialize RAG and page finder (cached across reruns)
rag, page_finder = init_rag()



# -----------------------------
# TITLE + DESCRIPTION
# -----------------------------
st.title("צ'אטבוט עיריית חיפה")

st.markdown("""
<div>
ברוכים הבאים לעוזר החכם של עיריית חיפה!  
שאלו אותי כל שאלה על שירותים עירוניים, תשלומים, חניה, ארנונה, חינוך, אירועים ועוד.

<br>

<b>דוגמאות:</b><br>
• מה העיר חיפה מציעה למבקרים בה?<br>
• איך מזמינים מגרש כדורסל?<br>
• איך מגישים בקשה להנחה בארנונה?<br>
</div>
""", unsafe_allow_html=True)



# -----------------------------
# CHAT DISPLAY
# -----------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        st.markdown(
            f'<div class="chat-bubble user-bubble"><strong>משתמש:</strong><br>{content}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-bubble assistant-bubble"><strong>תשובה:</strong><br>{content}</div>',
            unsafe_allow_html=True
        )



# -----------------------------
# CHAT INPUT
# -----------------------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("הזן הודעה:", "", placeholder="כתבו את השאלה שלכם כאן...")
    send = st.form_submit_button("שלח")

    if send and user_input.strip():
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            # Check if RAG system was initialized successfully
            if rag is None:
                answer = "שגיאה: המערכת לא אותחלה כראוי. אנא בדקו את קובץ המפתחות API ונסו לרענן את העמוד."
            else:
                # Display loading spinner while processing
                with st.spinner("מחפש מידע ומכין תשובה..."):
                    # Get answer from RAG system
                    # Using adaptive chunking strategy with k=10 for better retrieval
                    res = rag.answer_question(user_input, top_k=10, strategy="adaptive")
                    answer = res["answer"]
                    confidence = res.get("confidence", {})

                    # Add relevant page recommendations from Smart Page Finder
                    # This uses semantic similarity to find official Haifa municipality pages
                    if page_finder is not None:
                        pages = page_finder.find_relevant_pages(user_input)
                        if pages:
                            answer += "\n\n---\n\n**קישורים רלוונטיים:**\n"
                            for i, p in enumerate(pages, 1):
                                ttl = p["title"]
                                sub = p.get("subtitle", "")
                                url = p["url"]
                                # Format display: "Title - Subtitle" or just "Title"
                                display = f"{ttl} - {sub}" if sub else ttl
                                answer += f"{i}. [{display}]({url})\n"
                                
                    # Add confidence visualization to answer
                    # Confidence score is calculated based on:
                    # - Query-chunk similarity (50%)
                    # - Chunk agreement (30%)
                    # - Supported claims ratio (20%)
                    if confidence:
                        score = confidence.get("confidence_score", 0)
                        level = confidence.get("confidence_level", "Low")
                        reason = confidence.get("reason", "")

                        # Determine color and emoji based on confidence level
                        # High: ≥70% (green), Medium: 40-69% (yellow), Low: <40% (red)
                        if level == "High":
                            color = "#7CC242"; emoji = "🟢"
                        elif level == "Medium":
                            color = "#FFA500"; emoji = "🟡"
                        else:
                            color = "#FF6B6B"; emoji = "🔴"

                        # Build HTML for confidence meter visualization
                        # Includes: emoji indicator, score percentage, progress bar, and reason
                        confidence_html = f"""
                        <div style="margin-top:15px; padding:15px; background:#f8f9fa; border-radius:8px; border-right:4px solid {color};">
                            <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                                <span style="font-size:20px;">{emoji}</span>
                                <strong style="color:{color}; font-size:18px;">
                                    רמת ביטחון: {score}% ({level})
                                </strong>
                            </div>

                            <div style="background:#e0e0e0; border-radius:10px; height:20px; overflow:hidden; margin-bottom:10px;">
                                <div style="background:{color}; height:100%; width:{score}%;"></div>
                            </div>

                            <div style="font-size:14px; color:#555;">
                                <strong>סיבה:</strong> {reason}
                            </div>
                        </div>
                        """
                        answer += confidence_html

        except Exception as e:
            # Handle any errors during answer generation
            answer = f"שגיאה במהלך יצירת התשובה: {e}"

        # Add assistant response to chat history and rerun to display it
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()



# -----------------------------
# SIDEBAR — CHAT HISTORY
# -----------------------------
# Sidebar section for managing chat history and session statistics
st.sidebar.markdown("### היסטוריית צ'אט")

# Count number of user questions in current session
num_questions = sum(1 for m in st.session_state.messages if m["role"] == "user")
st.sidebar.write(f"מספר שאלות: {num_questions}")

# Button to clear chat history
# This removes all messages from session state and refreshes the page
if st.sidebar.button("נקה היסטוריה"):
    st.session_state.messages = []
    st.rerun()

# Button to export chat history as text file
# Formats all messages as "ROLE: content" and provides download
if st.sidebar.button("ייצא היסטוריה"):
    if st.session_state.messages:
        # Format messages for export: "USER: ..." or "ASSISTANT: ..."
        history_text = "\n\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in st.session_state.messages
        )
        # Provide download button with formatted history
        st.sidebar.download_button(
            "הורד",
            history_text,
            "chat_history.txt",
            "text/plain"
        )
    else:
        st.sidebar.info("אין היסטוריה לשמור")
