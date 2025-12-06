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

/* ----- FIX SIDEBAR ALWAYS OPEN ----- */
[data-testid="stSidebar"] {
    min-width: 260px !important;
    max-width: 260px !important;
}
button[kind="header"] {
    display: none !important;
}

/* ----- PAGE WIDTH ----- */
.block-container {
    max-width: 880px !important;
    margin: auto !important;
}

/* ----- TITLE + LOGO ROW ----- */
.header-row {
    display: flex;
    align-items: center;
    gap: 20px;
    justify-content: center;
    margin-bottom: 25px;
}
.header-title {
    font-size: 36px;
    font-weight: bold;
    color: #195c8c;
}

/* ----- CHAT BUBBLES ----- */
.chat-bubble {
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    line-height: 1.6;
}

.user-bubble {
    background-color: #e6f3ff;
    border-right: 6px solid #4FA3D1;
}

.assistant-bubble {
    background-color: #ffffff;
    border-right: 6px solid #7CC242;
}

/* Input alignment */
input, textarea {
    direction: rtl !important;
    text-align: right !important;
}

/* Buttons */
button {
    background-color: #195c8c !important;
    color: white !important;
    border-radius: 8px !important;
}
button:hover {
    background-color: #4fa3d1 !important;
}

/* Links */
a { color: #195c8c !important; font-weight: bold; }

</style>
""", unsafe_allow_html=True)



# -----------------------------
# HEADER: LOGO + TITLE SAME ROW
# -----------------------------
st.markdown("""
<div class="header-row">
    <img src="logos/logo2.png" width="200">
    <div class="header-title">צ'אטבוט עיריית חיפה</div>
</div>
""", unsafe_allow_html=True)



# -----------------------------
# INITIALIZE RAG
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
# DESCRIPTION
# -----------------------------
st.markdown("""
ברוכים הבאים לעוזר החכם של עיריית חיפה!  
שאלו אותי כל שאלה על שירותים עירוניים, תשלומים, חניה, ארנונה, חינוך, אירועים ועוד.

**דוגמאות:**
- מה העיר חיפה מציעה למבקרים בה?
- איך מזמינים מגרש כדורסל?
- איך מגישים בקשה להנחה בארנונה?
""")


# -----------------------------
# CHAT HISTORY DISPLAY
# -----------------------------
for msg in st.session_state.messages:

    # USER
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-bubble user-bubble"><strong>משתמש:</strong><br>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )

    # ASSISTANT (Split bubble + HTML content)
    else:
        st.markdown(
            f'<div class="chat-bubble assistant-bubble"><strong>עוזר:</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown(msg["content"], unsafe_allow_html=True)


# -----------------------------
# INPUT BOX
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
                pages = page_finder.find_relevant_pages(user_input)

                # append relevant page links
                if pages:
                    answer += "\n\n---\n\n**קישורים רלוונטיים:**\n"
                    for i, p in enumerate(pages, 1):
                        ttl = p["title"]
                        sub = p.get("subtitle", "")
                        url = p["url"]
                        display = f"{ttl} - {sub}" if sub else ttl
                        answer += f"{i}. [{display}]({url})\n"

                # CONFIDENCE widget (HTML block)
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
                        <div style="display:flex; align-items:center; gap:10px;">
                            <span style="font-size:20px;">{emoji}</span>
                            <strong style="color:{color}; font-size:18px;">
                                רמת ביטחון: {score}% ({level})
                            </strong>
                        </div>

                        <div style="background:#e0e0e0; border-radius:10px; height:20px; overflow:hidden; margin:10px 0;">
                            <div style="background:{color}; height:100%; width:{score}%;"></div>
                        </div>

                        <div style="font-size:14px; color:#555;"><strong>סיבה:</strong> {reason}</div>
                    </div>
                    """

                    answer += confidence_html

        except Exception as e:
            answer = f"שגיאה במהלך יצירת התשובה: {e}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()



# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.markdown("### היסטוריית צ'אט")
st.sidebar.write(f"מספר הודעות: {len(st.session_state.messages)}")

if st.sidebar.button("נקה היסטוריה"):
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("ייצא היסטוריה"):
    if st.session_state.messages:
        data = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages
        )
        st.sidebar.download_button("הורד", data, "chat_history.txt")
    else:
        st.sidebar.info("אין היסטוריה לשמור")
