import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import base64
from io import BytesIO
from langchain_core.messages import HumanMessage, AIMessage

def render_ui(
    suggestions,
    logo_path,
    get_response_streaming,
    memory,
):
    # Set fonts and logo
    st.set_page_config(page_title="עוזר וירטואלי AI - מידע עירוני", layout="wide")

    # Google Fonts and custom CSS
    components.html("""
    <link href="https://fonts.googleapis.com/css2?family=Assistant:wght@700&display=swap" rel="stylesheet">
    """, height=0)

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Assistant:wght@700&display=swap');
    body, .stApp, .stTextInput, .stTextArea { direction: rtl; }
    .custom-header { font-family: 'Assistant', sans-serif !important; font-weight: 700; color: black; margin: 0; text-align: center; }
    div.stButton > button {
        background-color: rgba(240, 244, 250, 0.6);
        color: #002B5B;
        border: 2px solid rgba(153, 209, 255, 0.4);
        border-radius: 18px;
        padding: 0.75rem 1rem;
        font-size: 14px;
        font-weight: bold;
        white-space: normal;
        min-height: 80px;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        transition: all 0.3s ease, transform 0.2s ease;
        margin-bottom: 8px;
    }
    div.stButton > button:hover {
        background-color: rgba(217, 236, 255, 0.9);
        border-color: rgba(102, 178, 255, 0.8);
        color: #0413fa;
        cursor: pointer;
        transform: translateY(-6px);
    }
    [data-testid="stChatInputTextArea"] textarea:focus {
        outline: 2px solid #24b0ff !important;
        border: 2px solid #24b0ff !important;
        box-shadow: 0 0 0 1.5px #24b0ff !important;
    }
    </style>
    """, unsafe_allow_html=True)
    

    # Logo and main title 
    def pil_to_base64(img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    try:
        logo = Image.open(logo_path)
        logo_b64 = pil_to_base64(logo)
        logo_html = f'<img src="data:image/png;base64,{logo_b64}" width="75" height="auto" style="margin:0;padding:0;"/>'
    except FileNotFoundError:
        logo_html = ""

    st.markdown(f"""
    <div style="display: flex; flex-direction: row-reverse; align-items: center; justify-content: center; gap: 20px; margin-bottom: 1rem;">
        <h2 class="custom-header" style="margin:0;">עוזר וירטואלי AI - מידע עירוני</h2>
        {logo_html}
    </div>
    """, unsafe_allow_html=True)

    st.caption("ניתן לשאול על שירותים עירוניים, מידע על חיפה ותל אביב, רישיונות, תשלומים ועוד...")

    # Suggestion buttons, all in a single row if possible
    # st.markdown("###### לתקן לינקים ולהוסיף אותם לקונטקסט שהמודל מקבל")


    cols = st.columns(len(suggestions))
    if "user_input" not in st.session_state:
        st.session_state.user_input = None
    if "user_input_from_button" not in st.session_state:
        st.session_state.user_input_from_button = False

    for idx, suggestion in enumerate(suggestions):
        with cols[idx]:
            if st.button(suggestion, key=f"sugg_{idx}"):
                st.session_state.user_input = suggestion
                st.session_state.user_input_from_button = True
    
    for msg in memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)
    # Chat input (Hebrew placeholder)
    user_input = st.chat_input("הקלידו שאלה...")

    if user_input or st.session_state.get("user_input"):
        real_input = user_input if user_input else st.session_state["user_input"]
        st.session_state["user_input"] = None  # Reset after use

        # Display user message in chat
        with st.chat_message("user"):
            st.write(real_input)

        # Display assistant response with streaming
        with st.spinner("חושב..."):
            with st.chat_message("assistant"):
                container = st.empty()
                full_reply = ""
                for chunk in get_response_streaming(real_input, memory, k=7):
                    full_reply += chunk
                    # You can escape markdown if needed, e.g., full_reply.replace("$", "\\$")
                    container.markdown(full_reply)
                    

