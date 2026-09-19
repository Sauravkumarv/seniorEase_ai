import streamlit as st
import requests
import base64
import io
import pypdf

# Backend REST API Endpoints
BACKEND_URL = "http://localhost:5000/api/chat"
EXPLAIN_URL = "http://localhost:5000/api/explain"
IMAGE_URL = "http://localhost:5000/api/analyze-image"
DOC_URL = "http://localhost:5000/api/analyze-doc"
DOC_UPLOAD_URL = "http://localhost:5000/api/document/upload"
TTS_URL = "http://localhost:5000/api/tts"
VOICE_TRANSCRIBE_URL = "http://localhost:5000/api/voice/transcribe"
VOICE_SPEAK_URL = "http://localhost:5000/api/voice/speak"

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

    /* Tabs styling for clear separation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 21px !important;
        font-weight: 700 !important;
        padding: 12px 20px !important;
        border-radius: 12px 12px 0 0 !important;
        background-color: #E2E8F0 !important;
        color: #1E293B !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
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
    div.stButton > button[key="ask_button"], div.stButton > button[key="explain_button"], div.stButton > button[key="img_button"], div.stButton > button[key="doc_button"] {
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

    /* Secondary Utility Buttons */
    div.stButton > button[key="new_q_btn"], div.stButton > button[key="clear_conv_btn"] {
        background-color: #F1F5F9 !important;
        color: #334155 !important;
        font-size: 19px !important;
        font-weight: 600 !important;
        border: 2px solid #94A3B8 !important;
    }

    /* Large Textarea & File Uploader Styling */
    .stTextArea textarea {
        font-size: 22px !important;
        line-height: 1.6 !important;
        padding: 16px !important;
        border-radius: 14px !important;
        border: 2px solid #94A3B8 !important;
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    .stTextArea label, .stFileUploader label {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #1E293B !important;
        margin-bottom: 8px !important;
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

if "active_audio" not in st.session_state:
    st.session_state.active_audio = {}

if "voice_transcription" not in st.session_state:
    st.session_state.voice_transcription = ""

if "voice_status" not in st.session_state:
    st.session_state.voice_status = "idle"  # 'idle', 'transcribed', 'submitted', 'unclear'

if "voice_processed_audio_id" not in st.session_state:
    st.session_state.voice_processed_audio_id = None

# Helper function to request Audio TTS from backend (/api/voice/speak)
def play_audio_response(msg_idx: int, text_content: str, selected_lang: str):
    try:
        payload = {"text": text_content, "language": selected_lang}
        response = requests.post(VOICE_SPEAK_URL, json=payload, timeout=12)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                audio_b64 = data.get("audio_b64")
                st.session_state.active_audio[msg_idx] = base64.b64decode(audio_b64)
            else:
                st.error("Audio synthesis failed.")
        else:
            st.error("Unable to generate spoken audio.")
    except Exception as e:
        st.error(f"TTS connection error: {e}")

# Function to submit voice audio to /api/voice/transcribe REST endpoint
def submit_voice_transcription(audio_bytes: bytes, selected_lang: str):
    if not audio_bytes:
        return

    try:
        files = {'file': ('recording.wav', audio_bytes, 'audio/wav')}
        data = {'language': selected_lang}
        response = requests.post(VOICE_TRANSCRIBE_URL, files=files, data=data, timeout=15)

        if response.status_code == 200:
            res_data = response.json()
            if res_data.get("success"):
                transcribed_text = res_data.get("text", "").strip()
                if transcribed_text:
                    st.session_state.input_box_value = transcribed_text
                    submit_question(transcribed_text, selected_lang)
                else:
                    st.session_state.error_message = "Unable to understand voice recording. Please speak clearly or type your question."
            else:
                st.session_state.error_message = res_data.get("message", "Unable to understand voice recording.")
        else:
            st.session_state.error_message = "Unable to understand voice recording. Please speak clearly or type your question."
    except Exception:
        st.session_state.error_message = "Unable to connect to voice transcription service. Please try typing your question."

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

# Function to submit image for analysis
def submit_image_analysis(image_b64: str, mime_type: str, prompt_text: str, selected_lang: str):
    st.session_state.conversation_history.append({
        "role": "user",
        "content": f"🖼️ Photo Analysis Request:\n{prompt_text if prompt_text else 'Please inspect and explain this photo.'}"
    })

    try:
        payload = {
            "image_base64": image_b64,
            "mime_type": mime_type,
            "prompt": prompt_text,
            "language": selected_lang
        }
        response = requests.post(IMAGE_URL, json=payload, timeout=16)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                analysis = data.get("response", "Image analysis completed.")
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": analysis
                })
        else:
            st.session_state.error_message = (
                "Unable to connect to the assistant. Please make sure the Flask server is running."
            )

    except requests.exceptions.RequestException:
        st.session_state.error_message = (
            "Unable to connect to the assistant. Please make sure the Flask server is running."
        )

# Function to submit document question for analysis
def submit_doc_analysis(doc_id: str, doc_name: str, user_question: str, selected_lang: str, doc_text: str = ""):
    st.session_state.conversation_history.append({
        "role": "user",
        "content": f"📄 Document Question ({doc_name}):\n{user_question if user_question else 'Please summarize and explain this document.'}"
    })

    try:
        payload = {
            "document_id": doc_id,
            "document_text": doc_text,
            "question": user_question,
            "language": selected_lang
        }
        response = requests.post(DOC_URL, json=payload, timeout=14)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                doc_analysis = data.get("response", "Document analysis completed.")
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": doc_analysis
                })
            else:
                st.session_state.error_message = data.get("message", "Unable to analyze document.")
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

# 3. LANGUAGE SELECTOR
selected_language = st.radio(
    "Choose Language / भाषा चुनें:",
    options=["English", "Hindi", "Hinglish"],
    index=0,
    horizontal=True
)

st.write("") # Spacing

# 4. TAB NAVIGATION FOR FEATURES
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Ask & Voice Chat",
    "🖼️ Photo & Image Reader",
    "📄 Document Reader & Q&A",
    "📖 Explain Something"
])

# --- TAB 1: ASK & VOICE CHAT ---
with tab1:
    # 🎙️ VOICE MODE SECTION
    st.markdown('<div class="section-title">🎙️ Voice Mode</div>', unsafe_allow_html=True)
    st.info("💡 Speak your question aloud. We will transcribe your voice so you can review before sending.")

    # Controls: [Start Speaking] and Selected Language display
    rec_col, lang_col = st.columns([2, 1])
    with rec_col:
        recorded_audio = st.audio_input("Start Speaking (Record Voice)", key="voice_mode_recorder")
    with lang_col:
        st.markdown("**Language:**")
        st.write(f"🗣️ **{selected_language}**")

    # Audio Capture & Single-time Transcription Check
    if recorded_audio is not None:
        audio_bytes = recorded_audio.getvalue()
        audio_id = hash(audio_bytes)

        if st.session_state.voice_processed_audio_id != audio_id:
            with st.spinner("⏳ Converting your voice to text..."):
                try:
                    files = {'file': ('recording.wav', audio_bytes, 'audio/wav')}
                    data = {'language': selected_language}
                    response = requests.post(VOICE_TRANSCRIBE_URL, files=files, data=data, timeout=15)
                    if response.status_code == 200:
                        res_data = response.json()
                        if res_data.get("success") and res_data.get("text", "").strip():
                            st.session_state.voice_transcription = res_data.get("text").strip()
                            st.session_state.voice_status = "transcribed"
                        else:
                            st.session_state.voice_status = "unclear"
                            st.session_state.voice_transcription = ""
                    else:
                        st.session_state.voice_status = "unclear"
                        st.session_state.voice_transcription = ""
                except Exception:
                    st.session_state.voice_status = "unclear"
                    st.session_state.voice_transcription = ""

                st.session_state.voice_processed_audio_id = audio_id
                st.rerun()

    # Did you mean? or Unclear Voice Warning
    if st.session_state.voice_status == "unclear":
        st.warning("⚠️ I couldn't hear you clearly. Please tap 'Start Speaking' and try repeating your question.")

    elif st.session_state.voice_status in ["transcribed", "submitted"] and st.session_state.voice_transcription:
        st.markdown(
            f"""
            <div style="background-color: #F0FDF4; border: 2px solid #22C55E; border-radius: 14px; padding: 18px 22px; margin-top: 14px; margin-bottom: 14px;">
                <div style="font-size: 20px; font-weight: 700; color: #15803D;">Did you mean?</div>
                <div style="font-size: 22px; color: #0F172A; font-weight: 600; margin-top: 6px;">"{st.session_state.voice_transcription}"</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # [Ask SeniorEase] Action Button
        if st.session_state.voice_status == "transcribed":
            if st.button("Ask SeniorEase", key="voice_mode_ask_btn"):
                submit_question(st.session_state.voice_transcription, selected_language)
                st.session_state.voice_status = "submitted"
                st.rerun()

        # Display AI Response & [🔊 Listen to Answer] button
        if st.session_state.voice_status == "submitted" and st.session_state.conversation_history:
            last_msg = st.session_state.conversation_history[-1]
            if last_msg["role"] == "assistant":
                st.markdown(
                    f"""
                    <div class="ai-card" style="margin-top: 14px;">
                        <div class="ai-card-title">👵 SeniorEase AI Response:</div>
                        <div class="card-content">{last_msg['content']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                last_idx = len(st.session_state.conversation_history) - 1
                if st.button("🔊 Listen to Answer", key="vm_listen_btn"):
                    play_audio_response(last_idx, last_msg["content"], selected_language)

                if last_idx in st.session_state.active_audio:
                    st.audio(st.session_state.active_audio[last_idx], format="audio/mp3")

                if st.button("🎙️ Speak Another Question", key="vm_reset_btn"):
                    st.session_state.voice_status = "idle"
                    st.session_state.voice_transcription = ""
                    st.session_state.voice_processed_audio_id = None
                    st.rerun()

    st.divider()
    st.markdown('<div class="section-title">💬 Type or Tap a Question</div>', unsafe_allow_html=True)

    # Quick Help Category Buttons
    st.markdown('**Tap a quick help topic below:**')

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

    user_question_input = st.text_area(
        label="Type your question here...",
        placeholder="Type your question here...",
        value=st.session_state.input_box_value,
        height=130,
        key="main_q_textarea"
    )

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

# --- TAB 2: PHOTO & IMAGE READER ---
with tab2:
    st.markdown('<div class="section-title">🖼️ Upload Photo / Image for Analysis</div>', unsafe_allow_html=True)
    st.info("💡 Upload photos of medicine labels, electricity bills, receipts, or official letters.")
    st.warning("🔒 Privacy Tip:\nOnly upload documents or images you are comfortable sharing with an AI service. Never upload passwords, OTPs, PINs or other secret credentials.")

    uploaded_image = st.file_uploader(
        "Select photo or image file:",
        type=["png", "jpg", "jpeg"],
        key="img_uploader"
    )

    image_prompt = st.text_area(
        label="What would you like to know about this photo?",
        placeholder="e.g., What is the dosage instruction? Or how much is the bill amount due?",
        height=100,
        key="img_prompt_area"
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image Preview", use_column_width=True)

        if st.button("Analyze Photo / Image", key="img_button"):
            with st.spinner("⏳ SeniorEase AI is inspecting your image..."):
                img_bytes = uploaded_image.getvalue()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                mime_type = uploaded_image.type or "image/jpeg"
                submit_image_analysis(img_b64, mime_type, image_prompt, selected_language)
                st.rerun()

# --- TAB 3: ASK ABOUT A DOCUMENT ---
with tab3:
    st.markdown('<div class="section-title">📄 Ask About a Document</div>', unsafe_allow_html=True)
    st.info("💡 Upload pension forms, bank statements, or official notices (PDF, DOCX, TXT).")
    st.warning("🔒 Privacy Tip:\nOnly upload documents or images you are comfortable sharing with an AI service. Never upload passwords, OTPs, PINs or other secret credentials.")

    if "current_doc_info" not in st.session_state:
        st.session_state.current_doc_info = None

    uploaded_doc = st.file_uploader(
        "Upload document:",
        type=["pdf", "docx", "txt"],
        key="doc_uploader"
    )

    if uploaded_doc is not None:
        doc_bytes = uploaded_doc.getvalue()
        audio_id = hash(doc_bytes)

        # Upload document to backend if not already uploaded in session_state
        if not st.session_state.current_doc_info or st.session_state.current_doc_info.get("hash") != audio_id:
            with st.spinner("⏳ Extracting text and indexing document..."):
                try:
                    files = {'file': (uploaded_doc.name, doc_bytes, uploaded_doc.type or 'application/octet-stream')}
                    res = requests.post(DOC_UPLOAD_URL, files=files, timeout=15)
                    if res.status_code == 200 and res.json().get("success"):
                        info = res.json()
                        info["hash"] = audio_id
                        st.session_state.current_doc_info = info
                    else:
                        st.session_state.current_doc_info = {"error": "Could not extract text from document."}
                except Exception as e:
                    st.session_state.current_doc_info = {"error": str(e)}

        doc_info = st.session_state.current_doc_info

        if doc_info and "document_id" in doc_info:
            # Metadata Display Box
            st.markdown(
                f"""
                <div style="background-color: #F0FDF4; border: 2px solid #22C55E; border-radius: 14px; padding: 18px 22px; margin-top: 14px; margin-bottom: 20px;">
                    <div style="font-size: 22px; font-weight: 800; color: #15803D; margin-bottom: 8px;">✓ Document uploaded</div>
                    <div style="font-size: 20px; color: #0F172A; margin-bottom: 4px;"><b>File name:</b> {doc_info['filename']}</div>
                    <div style="font-size: 20px; color: #0F172A; margin-bottom: 4px;"><b>Pages:</b> {doc_info['page_count']}</div>
                    <div style="font-size: 20px; color: #0F172A;"><b>Extracted text available:</b> Yes</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Question Input Area
            doc_question = st.text_area(
                label="What would you like to know?",
                placeholder="e.g., How much money was spent this month? Or what is the due date?",
                height=130,
                key="doc_question_area"
            )

            btn_col1, btn_col2 = st.columns([2, 1])

            with btn_col1:
                if st.button("Ask About Document", key="doc_button"):
                    if doc_question and doc_question.strip():
                        with st.spinner("⏳ SeniorEase AI is finding relevant details in your document..."):
                            submit_doc_analysis(
                                doc_id=doc_info["document_id"],
                                doc_name=doc_info["filename"],
                                user_question=doc_question,
                                selected_lang=selected_language
                            )
                            st.rerun()
                    else:
                        st.warning("Please type a question about your document.")

            with btn_col2:
                if st.button("Clear Document", key="clear_doc_btn", use_container_width=True):
                    st.session_state.current_doc_info = None
                    st.rerun()

        elif doc_info and "error" in doc_info:
            st.error(f"Document processing issue: {doc_info['error']}")

# --- TAB 4: EXPLAIN SOMETHING SIMPLY ---
with tab4:
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

# 9. CONVERSATION CARDS SECTION WITH AUDIO SPOKEN RESPONSE BUTTONS
if st.session_state.conversation_history:
    st.divider()
    
    header_col, clear_col = st.columns([2, 1])
    with header_col:
        st.markdown('<div class="section-title">💬 Your Conversation</div>', unsafe_allow_html=True)
    with clear_col:
        if st.button("🗑️ Clear Conversation", key="clear_conv_btn", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.input_box_value = ""
            st.session_state.active_audio = {}
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

            # Audio Spoken Voice Button for Elderly Users
            audio_btn_key = f"tts_btn_{idx}"
            if st.button("🔊 Listen to Answer (Spoken Audio)", key=audio_btn_key):
                play_audio_response(idx, msg["content"], selected_language)

            if idx in st.session_state.active_audio:
                st.audio(st.session_state.active_audio[idx], format="audio/mp3")
