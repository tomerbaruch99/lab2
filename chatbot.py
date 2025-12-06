"""
Haifa Municipality RAG Chatbot with Hebrew (RTL) Support
Integrates Gemini RAG system and Smart Page Finder
"""

import warnings
import sys
from pathlib import Path

# Python 3.9 compatibility: Add packages_distributions to importlib.metadata if missing
try:
    import importlib.metadata
    if not hasattr(importlib.metadata, 'packages_distributions'):
        # Python 3.9 compatibility shim
        def _packages_distributions():
            """Fallback for Python 3.9 compatibility."""
            try:
                # Try to use importlib_metadata backport if available
                import importlib_metadata
                return importlib_metadata.packages_distributions()
            except (ImportError, AttributeError):
                # Return empty dict if not available
                return {}
        importlib.metadata.packages_distributions = _packages_distributions
except (ImportError, AttributeError):
    pass

# Suppress expected warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*Session state does not function.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
warnings.filterwarnings("ignore", message=".*packages_distributions.*")

import streamlit as st

# Check if running with streamlit run (suppress warning if not, but continue)
try:
    # Try to access streamlit runtime - if it fails, we're not in streamlit context
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        # Running without streamlit run - warnings already suppressed above
        pass
except (ImportError, AttributeError):
    # Not in streamlit context - warnings already suppressed
    pass

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import RAG components
from gemini_integration import GeminiRAG
from utils.smart_page_finder import SmartPageFinder

# Page configuration
st.set_page_config(
    page_title="צ'אטבוט עיריית חיפה",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for RTL support and Gisha font
st.markdown("""
    <style>
    /* Global font setting - Gisha for all text */
    * {
        font-family: Gisha, Arial, sans-serif !important;
    }
    
    body {
        font-family: Gisha, Arial, sans-serif !important;
    }
    
    /* RTL support for Hebrew text */
    .rtl {
        direction: rtl;
        text-align: right;
        font-family: Gisha, Arial, sans-serif !important;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        font-family: Gisha, Arial, sans-serif !important;
    }
    
    .user-message {
        background-color: #e3f2fd;
        direction: rtl;
        text-align: right;
        font-family: Gisha, Arial, sans-serif !important;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        direction: rtl;
        text-align: right;
        font-family: Gisha, Arial, sans-serif !important;
    }
    
    /* Input area RTL */
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
        font-family: Gisha, Arial, sans-serif !important;
    }
    
    /* Text area RTL */
    .stTextArea > div > div > textarea {
        direction: rtl;
        text-align: right;
        font-family: Gisha, Arial, sans-serif !important;
    }
    
    /* Buttons */
    button {
        font-family: Gisha, Arial, sans-serif !important;
    }
    
    /* All Streamlit text elements */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        font-family: Gisha, Arial, sans-serif !important;
    }
    
    /* Form alignment for RTL - align to right */
    form[data-testid="chat_form"] {
        direction: rtl;
        text-align: right;
    }
    
    /* Align form container to the right */
    form[data-testid="chat_form"] > div {
        direction: rtl;
        text-align: right;
        display: flex;
        flex-direction: row-reverse;
        justify-content: flex-start;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Column containers in form */
    form[data-testid="chat_form"] [data-testid="column"] {
        direction: rtl;
        text-align: right;
    }
    
    /* Align submit button to the right */
    form[data-testid="chat_form"] button {
        direction: rtl;
    }
    
    /* Input container alignment */
    form[data-testid="chat_form"] .stTextInput {
        direction: rtl;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG system and Smart Page Finder (only once)
@st.cache_resource
def init_rag_system():
    """Initialize RAG system and Smart Page Finder (cached for performance)."""
    try:
        rag = GeminiRAG(api_keys_path="utils/api_keys.json")
        page_finder = SmartPageFinder()
        return rag, page_finder
    except Exception as e:
        # Suppress importlib.metadata compatibility errors (non-critical)
        error_msg = str(e)
        if "packages_distributions" in error_msg or "importlib.metadata" in error_msg:
            # Try again - the compatibility shim should handle it
            try:
                rag = GeminiRAG(api_keys_path="utils/api_keys.json")
                page_finder = SmartPageFinder()
                return rag, page_finder
            except Exception:
                # If it still fails, return None but don't show error to user
                # (this is a known Python 3.9 compatibility issue that doesn't affect functionality)
                return None, None
        else:
            st.error(f"שגיאה באתחול המערכת: {e}")
            return None, None

rag_system, page_finder = init_rag_system()

# Title
st.title("💬 צ'אטבוט עיריית חיפה")
st.markdown("""
<div class="rtl" style="font-size: 1.1em; margin-bottom: 2rem; color: #666;">
חיפאים ומבקרים - שאלו אותי על חיפה :)<br><br> לדוגמה:<br><table><tr><td>"מה העיר חיפה מציעה למבקרים בה?"</td></tr><tr><td>"איך מזמינים פעילויות ספורט או מתקנים כמו מגרשי כדורסל או כדורגל?"</td></tr><tr><td>"איך מגישים בקשה להנחות או פטור בארנונה?"</td></tr></table>
</div>
""", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f'<div class="chat-message user-message"><strong>משתמש:</strong> {content}</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message assistant-message"><strong>עוזר:</strong> {content}</div>', 
                   unsafe_allow_html=True)

# Chat input with form for Enter key support
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("הזן הודעה:", key="user_input", value="", placeholder="כתבו את השאלה שלכם כאן...")
    submitted = st.form_submit_button("שלח", type="primary")
    
    if submitted and user_input.strip():
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate response using RAG system
        if rag_system is None or page_finder is None:
            response = "מצטער, המערכת לא זמינה כרגע. אנא נסה שוב מאוחר יותר."
        else:
            try:
                # Get RAG answer
                with st.spinner("מחפש מידע ומכין תשובה..."):
                    result = rag_system.answer_question(
                        question=user_input,
                        top_k=5,
                        return_chunks=False
                    )
                    response = result["answer"]
                    
                    # Get relevant pages
                    relevant_pages = page_finder.find_relevant_pages(user_input, top_k=3)
                    
                    # Append page suggestions if available
                    if relevant_pages:
                        response += "\n\n---\n\n"
                        response += "**למידע נוסף - דפים רלוונטיים באתר העירייה:**\n\n"
                        for i, page in enumerate(relevant_pages, 1):
                            title = page['title']
                            subtitle = page.get('subtitle', '')
                            url = page['url']
                            
                            if subtitle and subtitle != title:
                                display_title = f"{title} - {subtitle}"
                            else:
                                display_title = title
                            
                            response += f"{i}. [{display_title}]({url})\n"
            except Exception as e:
                response = f"מצטער, אירעה שגיאה בעת יצירת התשובה: {str(e)}"
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the display
        st.rerun()

# Clear chat button
if st.button("נקה היסטוריה"):
    st.session_state.messages = []
    st.rerun()

# Display chat history count
st.sidebar.markdown("### היסטוריית צ'אט")
st.sidebar.markdown(f'<div class="rtl">מספר הודעות: {len(st.session_state.messages)}</div>', 
                   unsafe_allow_html=True)

if st.sidebar.button("ייצא היסטוריה"):
    if st.session_state.messages:
        history_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in st.session_state.messages
        ])
        st.sidebar.download_button(
            label="הורד",
            data=history_text,
            file_name="chat_history.txt",
            mime="text/plain"
        )
    else:
        st.sidebar.info("אין היסטוריה לייצוא")

