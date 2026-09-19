import streamlit as st
import requests

# Backend REST API Endpoints
BACKEND_URL = "http://localhost:5000/api/chat"
EXPLAIN_URL = "http://localhost:5000/api/explain"

# Streamlit Page Configuration - Clean, Centered, Professional
st.set_page_config(
    page_title="SeniorEase AI",
    page_icon="👵",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Senior-Accessible CSS System
CUSTOM_SENIOR_CSS = """
<style>
    /* Global Base Typography & Accessibility */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif !important;
        font-size: 22px !important;
        line-height: 1.7 !important;
        color: #0F172A !important;
        background-color: #FAFAFA;
    }

    /* Main Responsive Container Width & Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 850px !important;
    }

    /* Header Banner Styling - Dignified & Calm */
    .app-header {
        text-align: center;
        margin-bottom: 24px;
    }

    .app-title {
        font-size: 46px !important;
        font-weight: 800 !important;
        color: #1E3A8A !important; /* Deep Royal Navy */
        margin-bottom: 6px !important;
        letter-spacing: -0.5px;
    }

    .app-subtitle {
        font-size: 24px !important;
        color: #475569 !important; /* Slate Gray */
        font-weight: 500 !important;
        margin-bottom: 16px !important;
    }

    /* Section Headings */
    .section-title {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #1E293B !important;
        margin-top: 20px !important;
        margin-bottom: 14px !important;
    }

    /* Safety Tip Card - High Visibility & Reassuring */
    .safety-banner {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 24px;
        box-shadow: 0 2px 6px rgba(239, 68, 68, 0.08);
    }

    .safety-title {
        font-size: 22px !important;
        font-weight: 800 !important;
        color: #991B1B !important;
        margin-bottom: 6px !important;
    }

    .safety-text {
        font-size: 20px !important;
        color: #7F1D1D !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }

    /* Quick Help Section Container Card */
    .quick-help-card {
        background-color: #FFFFFF;
        border: 2px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.03);
    }

    /* Large Quick-Help Buttons Styling */
    div.stButton > button {
        font-size: 20px !important;
        font-weight: 600 !important;
        padding: 14px 18px !important;
        border-radius: 12px !important;
        border: 2px solid #CBD5E1 !important;
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        width: 100% !important;
        margin-bottom: 10px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
    }

    div.stButton > button:hover {
        border-color: #2563EB !important;
        background-color: #EFF6FF !important;
        color: #1E3A8A !important;
    }

    /* Primary Action Buttons */
    div.stButton > button[key="ask_button"], div.stButton > button[key="explain_button"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        padding: 15px 30px !important;
        border-radius: 14px !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
        margin-top: 10px !important;
        margin-bottom: 20px !important;
    }

    div.stButton > button[key="ask_button"]:hover, div.stButton > button[key="explain_button"]:hover {
        background-color: #1D4ED8 !important;
    }

    /* Secondary Utility Buttons */
    div.stButton > button[key="new_q_btn"], div.stButton > button[key="clear_conv_btn"] {
        background-color: #F1F5F9 !important;
        color: #334155 !important;
        font-size: 19px !important;
        font-weight: 600 !important;
        border: 2px solid #94A3B8 !important;
    }

    /* Large Textarea Styling */
    .stTextArea textarea {
        font-size: 22px !important;
        line-height: 1.6 !important;
        padding: 16px !important;
        border-radius: 14px !important;
        border: 2px solid #94A3B8 !important;
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    .stTextArea textarea:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.2) !important;
    }

    .stTextArea label {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #1E293B !important;
        margin-bottom: 8px !important;
    }

    /* Radio Selector Styling for Language */
    .stRadio label {
        font-size: 21px !important;
        font-weight: 600 !important;
        color: #1E293B !important;
    }

    /* User Question Card */
    .user-card {
        background-color: #EFF6FF;
        border-left: 8px solid #2563EB;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.06);
    }

    .user-card-title {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #1E40AF !important;
        margin-bottom: 8px !important;
    }

    /* AI Answer Card */
    .ai-card {
        background-color: #FFFFFF;
        border: 2px solid #E2E8F0;
        border-left: 8px solid #16A34A;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 28px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }

    .ai-card-title {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #15803D !important;
        margin-bottom: 10px !important;
    }

    .card-content {
        font-size: 22px !important;
        line-height: 1.7 !important;
        color: #0F172A !important;
        white-space: pre-wrap;
    }
</style>
"""

st.markdown(CUSTOM_SENIOR_CSS, unsafe_allow_html=True)

# Initialize Session State Variables
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "input_box_value" not in st.session_state:
    st.session_state.input_box_value = ""

# Function to submit chat question to Flask REST API
def submit_question(user_query: str, selected_lang: str, category_name: str = "general"):
    if not user_query or not user_query.strip():
        return

    query_text = user_query.strip()

    st.session_state.conversation_history.append({
        "role": "user",
        "content": query_text
    })

    try:
        payload = {
            "message": query_text,
            "language": selected_lang,
            "category": category_name
        }
        response = requests.post(BACKEND_URL, json=payload, timeout=12)

        if response.status_code == 200:
            data = response.json()
            ai_reply = data.get("response", "Thank you for asking. I am here to help you.")
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": ai_reply
            })
        else:
            st.session_state.error_message = (
                "Unable to connect to the assistant. Please make sure the Flask server is running."
            )

    except requests.exceptions.RequestException:
        st.session_state.error_message = (
            "Unable to connect to the assistant. Please make sure the Flask server is running."
        )

# Function to submit difficult text to /api/explain endpoint
def submit_explain(text_to_explain: str, selected_lang: str):
    if not text_to_explain or not text_to_explain.strip():
        return

    cleaned_text = text_to_explain.strip()

    st.session_state.conversation_history.append({
        "role": "user",
        "content": f"📖 Explain Simply:\n\n\"{cleaned_text}\""
    })

    try:
        payload = {
            "text": cleaned_text,
            "language": selected_lang
        }
        response = requests.post(EXPLAIN_URL, json=payload, timeout=12)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                explanation = data.get("response", "Simplified explanation completed.")
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": explanation
                })
            else:
                st.session_state.error_message = data.get("message", "Unable to simplify text.")
        else:
            st.session_state.error_message = (
                "Unable to connect to the assistant. Please make sure the Flask server is running."
            )

    except requests.exceptions.RequestException:
        st.session_state.error_message = (
            "Unable to connect to the assistant. Please make sure the Flask server is running."
        )

# 1. TOP HEADER SECTION
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">SeniorEase AI</div>
        <div class="app-subtitle">Your simple digital companion</div>
    </div>
    """,
    unsafe_allow_html=True
)

# 2. VISIBLE SAFETY TIP BANNER
st.markdown(
    """
    <div class="safety-banner">
        <div class="safety-title">🔒 Safety Tip</div>
        <div class="safety-text">Never share your OTP, PIN, password, CVV or banking credentials with anyone.</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# 3. MAIN SECTION: "How can I help you today?"
st.markdown('<div class="section-title">How can I help you today?</div>', unsafe_allow_html=True)

# LANGUAGE SELECTOR
selected_language = st.radio(
    "Choose Language / भाषा चुनें:",
    options=["English", "Hindi", "Hinglish"],
    index=0,
    horizontal=True
)

st.write("") # Spacing

# 4. QUICK HELP BUTTONS CARD CONTAINER
st.markdown('**Tap a quick help topic below:**')

# Updated Example Questions per Language & Topic
EXAMPLE_QUESTIONS = {
    "English": {
        "whatsapp": "How do I send a photo to someone on WhatsApp?",
        "banking": "How can I check my bank balance safely online?",
        "train": "How can I book a train ticket online?",
        "email": "How do I send an email with an attachment?",
        "shopping": "How do I safely order something online?",
        "gov": "How can I find the official website for a government service?"
    },
    "Hindi": {
        "whatsapp": "व्हाट्सएप पर किसी को फोटो कैसे भेजें?",
        "banking": "ऑनलाइन अपना बैंक बैलेंस सुरक्षित रूप से कैसे देखें?",
        "train": "ऑनलाइन ट्रेन का टिकट कैसे बुक करें?",
        "email": "फाइल या फोटो के साथ ईमेल (Attachment) कैसे भेजें?",
        "shopping": "ऑनलाइन सुरक्षित रूप से कोई सामान कैसे मंगवाएं?",
        "gov": "किसी सरकारी सेवा की आधिकारिक वेबसाइट कैसे खोजें?"
    },
    "Hinglish": {
        "whatsapp": "WhatsApp par kisi ko photo kaise bhejein?",
        "banking": "Online apna bank balance safe tareeke se kaise dekhein?",
        "train": "Online train ticket kaise book karein?",
        "email": "Attachment ke saath email kaise bhejein?",
        "shopping": "Online safe tareeke se koi saman kaise order karein?",
        "gov": "Kisi government service ki official website kaise khojein?"
    }
}

col1, col2 = st.columns(2)

with col1:
    if st.button("📱 WhatsApp", use_container_width=True):
        q = EXAMPLE_QUESTIONS[selected_language]["whatsapp"]
        st.session_state.input_box_value = q
        submit_question(q, selected_language, category_name="whatsapp")
        st.rerun()

    if st.button("💳 Banking", use_container_width=True):
        q = EXAMPLE_QUESTIONS[selected_language]["banking"]
        st.session_state.input_box_value = q
        submit_question(q, selected_language, category_name="banking")
        st.rerun()

    if st.button("🚆 Train Booking", use_container_width=True):
        q = EXAMPLE_QUESTIONS[selected_language]["train"]
        st.session_state.input_box_value = q
        submit_question(q, selected_language, category_name="train")
        st.rerun()

with col2:
    if st.button("📧 Email", use_container_width=True):
        q = EXAMPLE_QUESTIONS[selected_language]["email"]
        st.session_state.input_box_value = q
        submit_question(q, selected_language, category_name="email")
        st.rerun()

    if st.button("🛒 Online Shopping", use_container_width=True):
        q = EXAMPLE_QUESTIONS[selected_language]["shopping"]
        st.session_state.input_box_value = q
        submit_question(q, selected_language, category_name="shopping")
        st.rerun()

    if st.button("🏛️ Government Services", use_container_width=True):
        q = EXAMPLE_QUESTIONS[selected_language]["gov"]
        st.session_state.input_box_value = q
        submit_question(q, selected_language, category_name="gov")
        st.rerun()

st.write("") # Spacing

# 5. LARGE QUESTION INPUT AREA
user_question_input = st.text_area(
    label="Type your question here...",
    placeholder="Type your question here...",
    value=st.session_state.input_box_value,
    height=130
)

# 6. ACTION BUTTON TOOLBAR: ASK BUTTON & START NEW QUESTION
ask_col, action_col = st.columns([2, 1])

with ask_col:
    if st.button("Ask SeniorEase", key="ask_button"):
        if user_question_input and user_question_input.strip():
            submit_question(user_question_input, selected_language)
            st.session_state.input_box_value = ""
            st.rerun()
        else:
            st.warning("Please type a question or tap one of the quick help buttons above.")

with action_col:
    if st.button("✏️ Start New Question", key="new_q_btn", use_container_width=True):
        st.session_state.input_box_value = ""
        st.rerun()

st.divider()

# 7. FEATURE: EXPLAIN SOMETHING SIMPLY SECTION
st.markdown('<div class="section-title">📖 Explain Something Simply</div>', unsafe_allow_html=True)

explain_input_area = st.text_area(
    label="Paste any difficult message, notice or instruction here...",
    placeholder="Paste any difficult message, notice or instruction here...",
    height=130,
    key="explain_text_input"
)

if st.button("Explain Simply", key="explain_button"):
    if explain_input_area and explain_input_area.strip():
        submit_explain(explain_input_area, selected_language)
        st.rerun()
    else:
        st.warning("Please paste some text above to explain.")

# 8. BACKEND ERROR DISPLAY
if "error_message" in st.session_state and st.session_state.error_message:
    st.error(st.session_state.error_message)
    del st.session_state["error_message"]

# 9. CONVERSATION CARDS SECTION
if st.session_state.conversation_history:
    st.divider()
    
    header_col, clear_col = st.columns([2, 1])
    with header_col:
        st.markdown('<div class="section-title">💬 Your Conversation</div>', unsafe_allow_html=True)
    with clear_col:
        if st.button("🗑️ Clear Conversation", key="clear_conv_btn", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.input_box_value = ""
            st.rerun()

    for idx, msg in enumerate(st.session_state.conversation_history):
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="user-card">
                    <div class="user-card-title">👴 Your Question:</div>
                    <div class="card-content">{msg['content']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="ai-card">
                    <div class="ai-card-title">👵 SeniorEase AI Response:</div>
                    <div class="card-content">{msg['content']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
