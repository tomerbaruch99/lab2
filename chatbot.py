"""
Minimal Streamlit Chatbot with Hebrew (RTL) Support and Chat History
"""

import streamlit as st

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
        
        # Simple echo response (you can replace this with actual AI logic)
        response = f"קיבלתי את ההודעה שלך: {user_input}"
        
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

