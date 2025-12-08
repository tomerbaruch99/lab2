"""
Haifa Municipality RAG Chatbot
"""

import warnings
import sys
from pathlib import Path
import streamlit as st

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gemini_integration import GeminiRAG
from utils.smart_page_finder import SmartPageFinder



# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="צ'אטבוט עיריית חיפה",
    page_icon="logos/logo1.png",
    layout="centered"
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
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def init_rag():
    try:
        return GeminiRAG(api_keys_path="utils/api_keys.json"), SmartPageFinder()
    except:
        return None, None

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

        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            with st.spinner("מחפש מידע ומכין תשובה..."):

                res = rag.answer_question(user_input, top_k=5)
                answer = res["answer"]
                confidence = res.get("confidence", {})

                # relevant pages
                pages = page_finder.find_relevant_pages(user_input)
                if pages:
                    answer += "\n\n---\n\n**קישורים רלוונטיים:**\n"
                    for i, p in enumerate(pages, 1):
                        ttl = p["title"]
                        sub = p.get("subtitle", "")
                        url = p["url"]
                        display = f"{ttl} - {sub}" if sub else ttl
                        answer += f"{i}. [{display}]({url})\n"
                        
                # confidence visualization
                if confidence:
                    score = confidence.get("confidence_score", 0)
                    level = confidence.get("confidence_level", "Low")
                    reason = confidence.get("reason", "")

                    if level == "High":
                        color = "#7CC242"; emoji = "🟢"
                    elif level == "Medium":
                        color = "#FFA500"; emoji = "🟡"
                    else:
                        color = "#FF6B6B"; emoji = "🔴"

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
            answer = f"שגיאה במהלך יצירת התשובה: {e}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()



# -----------------------------
# SIDEBAR — CHAT HISTORY
# -----------------------------
st.sidebar.markdown("### היסטוריית צ'אט")

num_questions = sum(1 for m in st.session_state.messages if m["role"] == "user")
st.sidebar.write(f"מספר שאלות: {num_questions}")

if st.sidebar.button("נקה היסטוריה"):
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("ייצא היסטוריה"):
    if st.session_state.messages:
        history_text = "\n\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in st.session_state.messages
        )
        st.sidebar.download_button(
            "הורד",
            history_text,
            "chat_history.txt",
            "text/plain"
        )
    else:
        st.sidebar.info("אין היסטוריה לשמור")
