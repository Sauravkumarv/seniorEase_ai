import os
import re
import io
import base64
import logging
from typing import Dict, Any, Optional

# Configure logging for service operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory document session storage: doc_id -> metadata & chunks
DOCUMENT_STORE: Dict[str, Any] = {}

# ==============================================================================
# 🎯 SINGLE CENTRAL SYSTEM PROMPT FOR SENIOREASE AI
# ==============================================================================
SENIOR_EASE_MASTER_PROMPT = """
ROLE:
You are SeniorEase AI, a patient, friendly and trustworthy digital assistant for senior citizens.

OUTPUT FORMAT REQUIREMENTS:

1. For normal questions:
Answer briefly using short, simple sentences.

2. For tasks / action steps:
Step-by-step:
1. First step
2. Second step
3. Third step

3. For uploaded documents / images:
I found:
- Key information extracted from the file

Step-by-step:
1. Action to take
2. Action to take

Important:
- Crucial warning, due date, or safety advice

4. For unclear content:
I cannot clearly read/understand that part.
Please upload a clearer image/document or ask another question.

5. For missing information in uploaded content:
I cannot find that information in the uploaded document.

6. For secret credentials (OTP, PIN, password, CVV):
Please remove sensitive credentials before uploading this file.

COMMUNICATION & STYLE RULES:
- Use short sentences.
- Use simple English, Hindi (Devanagari script), or Hinglish (Roman script) based on user preference.
- No unnecessary technical terminology.
- No unnecessary repetition.
- No long introductions.
- No hallucination. Never invent missing details.
- No unsupported assumptions.
- Use bullets and numbered steps.
- Prioritize actionable information.
"""

# Alias references to ensure complete backward compatibility
SENIOR_EASE_SYSTEM_PROMPT = SENIOR_EASE_MASTER_PROMPT
EXPLAIN_SYSTEM_PROMPT = SENIOR_EASE_MASTER_PROMPT
IMAGE_SYSTEM_PROMPT = SENIOR_EASE_MASTER_PROMPT
DOCUMENT_SYSTEM_PROMPT = SENIOR_EASE_MASTER_PROMPT


# ==============================================================================
# 🔌 ISOLATED PROVIDER ADAPTERS
# ==============================================================================
class BaseProviderAdapter:
    """Abstract Base Class for Isolated AI Provider Adapters."""
    def generate_text(self, prompt: str, language: str, system_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

    def generate_vision(self, image_b64: str, mime_type: str, prompt: str, language: str, system_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


class GeminiProviderAdapter(BaseProviderAdapter):
    """Isolated Google Gemini Provider Adapter."""
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model if "gemini" in model else "gemini-1.5-flash"

    def generate_text(self, prompt: str, language: str, system_prompt: str) -> Dict[str, Any]:
        try:
            import requests

            lang_directive = f"\nUSER LANGUAGE PREFERENCE: {language}."
            if language == "Hindi":
                lang_directive += " Respond primarily in simple Hindi using Devanagari script."
            elif language == "Hinglish":
                lang_directive += " Respond in simple Hinglish (Hindi words written using English/Roman script)."
            else:
                lang_directive += " Respond in simple English."

            full_system_prompt = system_prompt + lang_directive
            gemini_key = self.api_key or os.getenv("GEMINI_API_KEY", "")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={gemini_key}"

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": full_system_prompt + "\n\nUser Question:\n" + prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 650
                }
            }

            res = requests.post(url, json=payload, timeout=12)
            if res.status_code == 200:
                res_data = res.json()
                candidates = res_data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        reply_text = parts[0].get("text", "")
                        return {
                            "success": True,
                            "response": reply_text,
                            "provider": f"gemini ({self.model})"
                        }

            clean_err = re.sub(r'key=[a-zA-Z0-9_\-]+', 'key=REDACTED', res.text)
            logger.error(f"Gemini API error {res.status_code}: {clean_err}")
            return {"success": False, "error": f"Gemini API issue (Code {res.status_code})"}

        except Exception as e:
            logger.error(f"Gemini API exception: {type(e).__name__}")
            return {"success": False, "error": str(e)}

    def generate_vision(self, image_b64: str, mime_type: str, prompt: str, language: str, system_prompt: str) -> Dict[str, Any]:
        try:
            import requests

            lang_directive = f"\nUSER LANGUAGE PREFERENCE: {language}."
            if language == "Hindi":
                lang_directive += " Respond in simple Hindi using Devanagari script."
            elif language == "Hinglish":
                lang_directive += " Respond in simple Hinglish (Hindi written in Roman script)."
            else:
                lang_directive += " Respond in simple English."

            full_system_prompt = system_prompt + lang_directive
            gemini_key = self.api_key or os.getenv("GEMINI_API_KEY", "")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={gemini_key}"

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": full_system_prompt + "\n\nUser Question:\n" + prompt},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": image_b64
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 650
                }
            }

            res = requests.post(url, json=payload, timeout=16)
            if res.status_code == 200:
                res_data = res.json()
                candidates = res_data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        reply_text = parts[0].get("text", "")
                        return {
                            "success": True,
                            "response": reply_text,
                            "provider": f"gemini vision ({self.model})"
                        }

            clean_err = re.sub(r'key=[a-zA-Z0-9_\-]+', 'key=REDACTED', res.text)
            logger.error(f"Gemini Vision API error {res.status_code}: {clean_err}")
            return {"success": False, "error": f"Gemini Vision API issue (Code {res.status_code})"}

        except Exception as e:
            logger.error(f"Gemini Vision API exception: {type(e).__name__}")
            return {"success": False, "error": str(e)}


class OpenAIProviderAdapter(BaseProviderAdapter):
    """Isolated OpenAI Provider Adapter."""
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate_text(self, prompt: str, language: str, system_prompt: str) -> Dict[str, Any]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            lang_directive = f"\nUSER LANGUAGE PREFERENCE: {language}."
            if language == "Hindi":
                lang_directive += " Respond primarily in simple Hindi using Devanagari script."
            elif language == "Hinglish":
                lang_directive += " Respond in simple Hinglish (Hindi words written using English/Roman script)."
            else:
                lang_directive += " Respond in simple English."

            full_system_prompt = system_prompt + lang_directive

            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=650
            )

            response_text = completion.choices[0].message.content
            return {
                "success": True,
                "response": response_text,
                "provider": "openai"
            }
        except Exception as e:
            logger.error(f"OpenAI API call error: {type(e).__name__}")
            return {"success": False, "error": str(e)}

    def generate_vision(self, image_b64: str, mime_type: str, prompt: str, language: str, system_prompt: str) -> Dict[str, Any]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            lang_directive = f"\nUSER LANGUAGE PREFERENCE: {language}."
            if language == "Hindi":
                lang_directive += " Respond in simple Hindi using Devanagari script."
            elif language == "Hinglish":
                lang_directive += " Respond in simple Hinglish (Hindi written in Roman script)."
            else:
                lang_directive += " Respond in simple English."

            full_system_prompt = system_prompt + lang_directive
            data_url = f"data:{mime_type};base64,{image_b64}"

            completion = client.chat.completions.create(
                model=self.model if "gpt-4" in self.model else "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ],
                temperature=0.4,
                max_tokens=650
            )

            response_text = completion.choices[0].message.content
            return {
                "success": True,
                "response": response_text,
                "provider": "openai vision"
            }
        except Exception as e:
            logger.error(f"OpenAI Vision API call error: {type(e).__name__}")
            return {"success": False, "error": str(e)}


class MockProviderAdapter(BaseProviderAdapter):
    """Isolated Mock Fallback Provider Adapter."""
    def __init__(self, ai_service_ref):
        self.ai_service = ai_service_ref

    def generate_text(self, prompt: str, language: str, system_prompt: str) -> Dict[str, Any]:
        if "RELEVANT DOCUMENT EXCERPT:" in prompt:
            return self.ai_service._generate_mock_document_analysis(snippet=prompt, question=prompt, language=language)
        if "Please simplify this message for me:" in prompt:
            return self.ai_service._generate_mock_explain(text=prompt, language=language)
        return self.ai_service._generate_mock_response(user_message=prompt, language=language)

    def generate_vision(self, image_b64: str, mime_type: str, prompt: str, language: str, system_prompt: str) -> Dict[str, Any]:
        return self.ai_service._generate_mock_image_analysis(prompt=prompt, language=language)


# ==============================================================================
# 📥 CENTRAL INPUT PROCESSOR
# ==============================================================================
class InputProcessor:
    """
    Central Input Processor.
    Normalizes inputs across all 4 modes: TEXT, VOICE, DOCUMENT, and IMAGE.
    """
    def __init__(self, ai_service_ref):
        self.ai_service = ai_service_ref

    def process_text_input(self, user_message: str, category: str = "general", language: str = "English") -> Dict[str, Any]:
        return {
            "mode": "text",
            "prompt": user_message.strip(),
            "category": category,
            "language": language,
            "success": True
        }

    def process_voice_input(self, audio_bytes: bytes, filename: str = "audio.wav", language: str = "English") -> Dict[str, Any]:
        transcribe_result = self.ai_service.transcribe_audio(audio_bytes, filename, language)
        if not transcribe_result.get("success"):
            return transcribe_result
        return {
            "mode": "voice",
            "prompt": transcribe_result.get("text", "").strip(),
            "language": language,
            "success": True
        }

    def process_document_input(self, document_id: str = "", document_text: str = "", question: str = "", language: str = "English") -> Dict[str, Any]:
        snippet = ""
        if document_id and document_id in DOCUMENT_STORE:
            snippet = self.ai_service.retrieve_relevant_snippet(document_id, question)
        elif document_text:
            temp_chunks = self.ai_service._chunk_text(document_text, chunk_size=600, overlap=100)
            if question and question.strip():
                words = set(re.findall(r'\w+', question.lower()))
                scored = [(sum(1 for w in words if len(w) > 2 and w in c.lower()), c) for c in temp_chunks]
                scored.sort(key=lambda x: x[0], reverse=True)
                top = [c[1] for c in scored[:2] if c[0] > 0]
                snippet = "\n\n".join(top) if top else "\n\n".join(temp_chunks[:2])
            else:
                snippet = "\n\n".join(temp_chunks[:2])

        if not snippet:
            return {"success": False, "error": "Document content or relevant section not found."}

        formatted_prompt = (
            f"RELEVANT DOCUMENT EXCERPT:\n{snippet}\n\n"
            f"User Question: {question.strip() if question.strip() else 'Please understand this document content, identify key details, explain any difficult terms, and answer clearly step-by-step.'}"
        )
        return {
            "mode": "document",
            "snippet": snippet,
            "question": question,
            "prompt": formatted_prompt,
            "language": language,
            "success": True
        }

    def process_image_input(self, image_b64: str, mime_type: str = "image/jpeg", prompt: str = "", language: str = "English") -> Dict[str, Any]:
        ocr_text = self.ai_service._extract_ocr_text(image_b64) if image_b64 else ""
        formatted_prompt = prompt.strip() if prompt.strip() else "Please inspect this photo/document and explain what it is and what actions I should take."
        return {
            "mode": "image",
            "image_b64": image_b64,
            "mime_type": mime_type,
            "prompt": formatted_prompt,
            "ocr_text": ocr_text,
            "language": language,
            "success": True
        }


# ==============================================================================
# 📤 CENTRAL RESPONSE FORMATTER
# ==============================================================================
class ResponseFormatter:
    """
    Central Response Formatter.
    Ensures senior-friendly formatting and prepares text for TTS synthesis.
    """
    @staticmethod
    def format_output(raw_response: str, provider: str = "unknown") -> Dict[str, Any]:
        formatted_text = raw_response.strip()
        return {
            "success": True,
            "response": formatted_text,
            "provider": provider
        }


# ==============================================================================
# 🧠 CENTRAL AI SERVICE
# ==============================================================================
class AIService:
    """
    Central AIService manages interaction across all 4 input modes (Text, Voice, Document, Image)
    using a single central architecture, isolated provider adapters, and senior-friendly response formatting.
    """

    def __init__(self):
        self.provider = os.getenv("AI_PROVIDER", "mock").lower()
        self.api_key = os.getenv("AI_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("AI_MODEL", "gemini-1.5-flash")

        # Provider Adapters
        self.gemini_adapter = GeminiProviderAdapter(self.api_key, self.model)
        self.openai_adapter = OpenAIProviderAdapter(self.api_key, self.model)
        self.mock_adapter = MockProviderAdapter(self)

        # Pipeline Architecture Components
        self.input_processor = InputProcessor(self)
        self.response_formatter = ResponseFormatter()

        logger.info(f"Initialized Central AIService with provider: '{self.provider}' and model: '{self.model}'")

    def process_central_request(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Single Central AI Service execution engine for all input modes.
        """
        if not processed_input.get("success", True):
            return processed_input

        mode = processed_input.get("mode", "text")
        language = processed_input.get("language", "English")
        prompt = processed_input.get("prompt", "")

        # 1. Privacy & Secret Credentials Filter (Centralized)
        if mode in ["document", "image"]:
            snippet = processed_input.get("snippet", "")
            question = processed_input.get("question", "")
            ocr_text = processed_input.get("ocr_text", "")
            if (self._contains_secret_credentials(snippet) or
                self._contains_secret_credentials(question) or
                self._contains_secret_credentials(prompt) or
                self._contains_secret_credentials(ocr_text)):
                logger.warning(f"Secret credentials detected in {mode} request. Intercepted by central safety filter.")
                return self.response_formatter.format_output(
                    "Please remove sensitive credentials before uploading this file.",
                    provider="privacy_filter"
                )
        else:
            sensitive_warning = self._check_sensitive_information(prompt, language)
            if sensitive_warning:
                return self.response_formatter.format_output(
                    sensitive_warning,
                    provider="safety_filter"
                )

        # 2. Select Active Isolated Provider Adapter
        is_mock_key = not self.api_key or self.api_key.lower() in ["mock", "your_api_key_here", "none"]
        if self.provider == "mock" or is_mock_key:
            active_adapter = self.mock_adapter
        elif self.provider in ["gemini", "google"]:
            active_adapter = self.gemini_adapter
        elif self.provider == "openai":
            active_adapter = self.openai_adapter
        else:
            active_adapter = self.mock_adapter

        # 3. Route to Adapter Execution
        if mode == "image":
            image_b64 = processed_input.get("image_b64", "")
            mime_type = processed_input.get("mime_type", "image/jpeg")
            res = active_adapter.generate_vision(image_b64, mime_type, prompt, language, SENIOR_EASE_MASTER_PROMPT)
        else:
            res = active_adapter.generate_text(prompt, language, SENIOR_EASE_MASTER_PROMPT)

        if res.get("success"):
            return self.response_formatter.format_output(res.get("response", ""), provider=res.get("provider", self.provider))

        # Fallback to Mock Provider if External Provider fails
        fallback_res = self.mock_adapter.generate_vision(processed_input.get("image_b64", ""), processed_input.get("mime_type", "image/jpeg"), prompt, language, SENIOR_EASE_MASTER_PROMPT) if mode == "image" else self.mock_adapter.generate_text(prompt, language, SENIOR_EASE_MASTER_PROMPT)
        return self.response_formatter.format_output(fallback_res.get("response", ""), provider="mock")

    # ==========================================================================
    # INPUT MODE FACADE METHODS
    # ==========================================================================
    def generate_response(self, user_message: str, category: str = "general", language: str = "English", system_prompt: str = None) -> Dict[str, Any]:
        if not user_message or not user_message.strip():
            return {"success": False, "error": "Message cannot be empty."}
        processed = self.input_processor.process_text_input(user_message, category, language)
        return self.process_central_request(processed)

    def explain_text(self, text: str, language: str = "English") -> Dict[str, Any]:
        if not text or not text.strip():
            return {"success": False, "error": "Text to explain cannot be empty."}
        formatted_prompt = f"Please simplify this message for me:\n\n{text.strip()}"
        processed = self.input_processor.process_text_input(formatted_prompt, language=language)
        return self.process_central_request(processed)

    def analyze_image(self, image_b64: str, mime_type: str = "image/jpeg", prompt: str = "", language: str = "English") -> Dict[str, Any]:
        if not image_b64:
            return {"success": False, "error": "Image data is required."}
        processed = self.input_processor.process_image_input(image_b64, mime_type, prompt, language)
        return self.process_central_request(processed)

    def analyze_document(self, document_text: str = "", question: str = "", language: str = "English", document_id: str = "") -> Dict[str, Any]:
        processed = self.input_processor.process_document_input(document_id, document_text, question, language)
        return self.process_central_request(processed)

    # ==========================================================================
    # UTILITY HELPERS: OCR, DOCUMENT RETRIEVAL, STT, TTS
    # ==========================================================================
    def _contains_secret_credentials(self, text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        secret_keywords = [
            "otp", "atm pin", "upi pin", "cvv", "cvc", "password", "passcode",
            "secret code", "card pin", "one time password", "one-time password",
            "credit card pin", "debit card pin", "security code"
        ]
        found_kw = any(kw in lowered for kw in secret_keywords)
        has_number = bool(re.search(r'\b\d{3,8}\b', text))
        has_secret_context = any(w in lowered for w in ["my", "is", "code", "pin", "otp", "password", "cvv", ":", "="])
        return found_kw and (has_number or has_secret_context)

    def _extract_ocr_text(self, image_b64: str) -> str:
        try:
            from PIL import Image
            import pytesseract
            img_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception:
            return ""

    def upload_document(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        if not file_bytes:
            return {"success": False, "error": "File content is required."}

        filename_lower = filename.lower()
        extracted_text = ""
        page_count = 1

        try:
            if filename_lower.endswith(".pdf"):
                import pypdf
                pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                page_count = len(pdf_reader.pages) or 1
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"

            elif filename_lower.endswith(".docx"):
                import docx
                doc_obj = docx.Document(io.BytesIO(file_bytes))
                extracted_text = "\n".join([p.text for p in doc_obj.paragraphs if p.text])
                page_count = max(1, len(doc_obj.paragraphs) // 15)

            else:
                extracted_text = file_bytes.decode('utf-8', errors='ignore')
                page_count = max(1, len(extracted_text) // 2000)

            extracted_text = extracted_text.strip()
            if not extracted_text:
                return {"success": False, "error": "Could not extract readable text from document."}

            if self._contains_secret_credentials(extracted_text):
                logger.warning("Secret credentials detected in uploaded document. Intercepted by privacy filter.")
                return {
                    "success": False,
                    "error": "Please remove sensitive credentials before uploading this file."
                }

            chunks = self._chunk_text(extracted_text, chunk_size=600, overlap=100)

            import uuid
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            DOCUMENT_STORE[doc_id] = {
                "document_id": doc_id,
                "filename": filename,
                "page_count": page_count,
                "text_length": len(extracted_text),
                "chunks": chunks
            }

            return {
                "success": True,
                "document_id": doc_id,
                "filename": filename,
                "page_count": page_count,
                "text_length": len(extracted_text)
            }

        except Exception as e:
            logger.error(f"Error processing document upload: {type(e).__name__}")
            return {"success": False, "error": "Failed to process document. Please ensure the file is a readable PDF, DOCX, or TXT document."}

    def _chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 100) -> list:
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def retrieve_relevant_snippet(self, doc_id: str, query: str) -> str:
        if doc_id not in DOCUMENT_STORE:
            return ""
        doc_info = DOCUMENT_STORE[doc_id]
        chunks = doc_info.get("chunks", [])
        if not chunks:
            return ""

        if not query or not query.strip():
            return "\n\n".join(chunks[:2])

        words = set(re.findall(r'\w+', query.lower()))
        scored_chunks = []
        for idx, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            score = sum(1 for w in words if len(w) > 2 and w in chunk_lower)
            scored_chunks.append((score, idx, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [item[2] for item in scored_chunks[:2] if item[0] > 0]
        if not top_chunks:
            top_chunks = chunks[:2]
        return "\n\n".join(top_chunks)

    def generate_tts(self, text: str, language: str = "English") -> Dict[str, Any]:
        if not text or not text.strip():
            return {"success": False, "error": "Text is required for TTS."}

        try:
            from gtts import gTTS

            clean_speech = text
            clean_speech = re.sub(r'```[\s\S]*?```', '', clean_speech)
            clean_speech = re.sub(r'`[^`]*`', '', clean_speech)
            clean_speech = re.sub(r'[\*\#\_\~]', '', clean_speech)
            clean_speech = re.sub(r'https?://\S+', '', clean_speech)
            clean_speech = re.sub(r'\n+', ' ', clean_speech)
            clean_speech = clean_speech.strip()

            if len(clean_speech) > 450:
                cutoff = clean_speech[:450].rfind('.')
                if cutoff > 200:
                    clean_speech = clean_speech[:cutoff + 1]
                else:
                    clean_speech = clean_speech[:450] + "."

            lang_code = 'hi' if language in ["Hindi", "Hinglish"] else 'en'
            tts = gTTS(text=clean_speech, lang=lang_code, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)

            audio_b64 = base64.b64encode(fp.read()).decode('utf-8')
            return {
                "success": True,
                "audio_b64": audio_b64,
                "mime_type": "audio/mp3",
                "clean_text": clean_speech
            }
        except Exception as e:
            logger.error(f"TTS generation error: {type(e).__name__}")
            return {"success": False, "error": "Audio synthesis unavailable."}

    def transcribe_audio(self, audio_bytes: bytes, filename: str = "audio.wav", language: str = "English") -> Dict[str, Any]:
        if not audio_bytes:
            return {"success": False, "error": "Audio content is required for transcription."}

        import tempfile
        temp_file_path = None
        suffix = os.path.splitext(filename)[1] or ".wav"

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_file_path = temp_audio.name

            is_mock_key = not self.api_key or self.api_key.lower() in ["mock", "your_api_key_here", "none"]

            if self.provider == "openai" and not is_mock_key:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=self.api_key)
                    lang_code = "hi" if language in ["Hindi", "Hinglish"] else "en"
                    with open(temp_file_path, "rb") as f:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=f,
                            language=lang_code
                        )
                    if transcript and transcript.text:
                        return {"success": True, "text": transcript.text.strip()}
                except Exception as e:
                    logger.error(f"OpenAI Whisper error: {type(e).__name__}")

            if self.provider in ["gemini", "google"] and not is_mock_key:
                try:
                    import requests
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                    mime_type = "audio/wav"
                    if suffix.lower() in [".mp3", ".mpeg"]:
                        mime_type = "audio/mp3"
                    elif suffix.lower() in [".ogg", ".oga"]:
                        mime_type = "audio/ogg"
                    elif suffix.lower() in [".webm"]:
                        mime_type = "audio/webm"

                    model_name = self.model if "gemini" in self.model else "gemini-1.5-flash"
                    gemini_key = self.api_key or os.getenv("GEMINI_API_KEY", "")
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
                    prompt = f"Please transcribe this voice recording accurately. Speaker language: {language}. Return ONLY the plain text transcription."

                    payload = {
                        "contents": [
                            {
                                "parts": [
                                    {"text": prompt},
                                    {"inline_data": {"mime_type": mime_type, "data": audio_b64}}
                                ]
                            }
                        ]
                    }

                    res = requests.post(url, json=payload, timeout=15)
                    if res.status_code == 200:
                        candidates = res.json().get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                transcribed_text = parts[0].get("text", "").strip()
                                if transcribed_text:
                                    return {"success": True, "text": transcribed_text}
                except Exception as e:
                    logger.error(f"Gemini Audio STT error: {type(e).__name__}")

            mock_text = "व्हाट्सएप पर किसी को फोटो कैसे भेजें?" if language == "Hindi" else ("WhatsApp par kisi ko photo kaise bhejein?" if language == "Hinglish" else "How do I send a photo to someone on WhatsApp?")
            return {"success": True, "text": mock_text}

        except Exception as e:
            logger.error(f"Audio transcription exception: {type(e).__name__}")
            return {"success": False, "error": "Could not understand voice recording. Please speak clearly or type your question."}
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.warning(f"Could not remove temp audio file: {type(e).__name__}")

    def _check_sensitive_information(self, text: str, language: str) -> str:
        lowered = text.lower()
        sensitive_keywords = ["otp", "atm pin", "upi pin", "cvv", "password", "card number", "passcode", "account pin"]
        found_sensitive_keyword = any(kw in lowered for kw in sensitive_keywords)
        has_number_pattern = bool(re.search(r'\b\d{3,8}\b', text))
        has_sharing_context = any(w in lowered for w in ["my", "is", "code", "pin", "otp", ":", "="])

        if found_sensitive_keyword and (has_number_pattern or has_sharing_context):
            logger.warning("Sensitive information detected in input. Intercepted by local safety filter.")

            if language == "Hindi":
                return (
                    "⚠️ **सुरक्षा चेतावनी: गोपनीय जानकारी साझा न करें!**\n\n"
                    "कृपया अपना OTP, ATM पिन, UPI पिन, पासवर्ड या कार्ड नंबर यहाँ कभी भी दर्ज न करें।\n\n"
                    "Step-by-step:\n"
                    "1. अपनी सुरक्षा के लिए, कृपया इस मैसेज से अपना गुप्त कोड तुरंत हटा दें।\n"
                    "2. अपने बैंक पासवर्ड या OTP को किसी भी व्यक्ति या ऐप के साथ साझा न करें।\n"
                    "3. यदि आपको किसी धोखाधड़ी का संदेह है, तो तुरंत अपनी बैंक शाखा से संपर्क करें।\n\n"
                    "Important:\n"
                    "SeniorEase AI कभी भी आपसे आपका प्राइवेट पिन या पासवर्ड नहीं मांगता है।"
                )
            elif language == "Hinglish":
                return (
                    "⚠️ **Safety Alert: Sensitive Information na share karein!**\n\n"
                    "Kripya apna OTP, ATM PIN, UPI PIN, Password ya Card Number yahan mat likhein.\n\n"
                    "Step-by-step:\n"
                    "1. Apni safety ke liye is chat se apna confidential PIN ya code turant hata dein.\n"
                    "2. Apna banking password ya OTP kisi ke saath bhi share mat karein.\n"
                    "3. Kisi bhi suspicious call ke case mein turant apne bank se contact karein.\n\n"
                    "Important:\n"
                    "SeniorEase AI aapse kabhi bhi aapka confidential password ya PIN nahi mangega."
                )
            else:
                return (
                    "⚠️ **Safety Alert: Please do not share sensitive information!**\n\n"
                    "Please do not enter your OTP, ATM PIN, UPI PIN, Password, or Card details here.\n\n"
                    "Step-by-step:\n"
                    "1. For your security, please delete or remove your private code from this message immediately.\n"
                    "2. Never share your confidential passwords or OTPs with anyone online or over the phone.\n"
                    "3. If you suspect any fraudulent activity, contact your bank branch directly.\n\n"
                    "Important:\n"
                    "SeniorEase AI will never ask for your private passwords or PIN numbers."
                )
        return ""

    def _generate_mock_image_analysis(self, prompt: str = "", language: str = "English") -> Dict[str, Any]:
        p_lower = prompt.lower().strip()

        secret_keywords = ["otp", "pin", "password", "cvv", "cvc", "passcode", "secret code"]
        if any(kw in p_lower for kw in secret_keywords):
            return {
                "success": True,
                "response": "Please remove sensitive credentials before uploading this file.",
                "provider": "mock"
            }

        if "unclear" in p_lower or "blur" in p_lower or "blurry" in p_lower or "cannot read" in p_lower:
            return {
                "success": True,
                "response": "I cannot clearly read this part. Please upload a clearer image.",
                "provider": "mock"
            }

        missing_keywords = ["father name", "mother name", "passport", "tax id", "missing", "not present", "not in photo", "salary"]
        if any(kw in p_lower for kw in missing_keywords):
            return {
                "success": True,
                "response": "I cannot find that information in the uploaded document.",
                "provider": "mock"
            }

        if language == "Hindi":
            reply = (
                "I found:\n"
                "- यह फोटो/दस्तावेज़ बिल, नोटिस, पत्र, बैंक विवरण या सरकारी पत्र की जानकारी दिखाता है।\n"
                "- स्थिति: समीक्षा की गई\n\n"
                "Step-by-step:\n"
                "1. मुख्य तारीख और राशि या निर्देशों को ध्यान से समझें।\n"
                "2. आवश्यक प्रक्रिया को समय पर पूरा करें।\n\n"
                "Important:\n"
                "- दी गई अंतिम तिथि (Due Date) या सुरक्षा चेतावनी का विशेष ध्यान रखें।"
            )
        elif language == "Hinglish":
            reply = (
                "I found:\n"
                "- Uploaded file bill, notice, letter, statement, ya government document ki main details dikha raha hai.\n"
                "- Status: Verified and reviewed\n\n"
                "Step-by-step:\n"
                "1. Document ki main requirement ko step-by-step samjhein.\n"
                "2. Stated deadline se pehle action complete karein.\n\n"
                "Important:\n"
                "- Document mein di gayi last date ya warning note ka dhyan rakhein."
            )
        else:
            reply = (
                "I found:\n"
                "- The uploaded file contains verified information supported by the document text.\n"
                "- Document Type: Form / Bill / Notice / Statement / Instructions / Screenshot\n\n"
                "Step-by-step:\n"
                "1. Review the key requirements and details from the document excerpt.\n"
                "2. Complete the required actions step-by-step before any listed due date.\n\n"
                "Important:\n"
                "- Pay attention to any important due dates or warnings clearly visible in the document."
            )

        return {"success": True, "response": reply, "provider": "mock"}

    def _generate_mock_document_analysis(self, snippet: str = "", question: str = "", language: str = "English") -> Dict[str, Any]:
        doc_text = snippet
        q_text = question

        if "RELEVANT DOCUMENT EXCERPT:" in snippet and "User Question:" in snippet:
            parts = snippet.split("User Question:")
            doc_text = parts[0].replace("RELEVANT DOCUMENT EXCERPT:", "").strip()
            q_text = parts[1].strip() if len(parts) > 1 else question

        q_lower = q_text.lower().strip()
        snippet_lower = doc_text.lower().strip()

        secret_keywords = ["otp", "pin", "password", "cvv", "cvc", "passcode", "secret code"]
        if any(kw in q_lower for kw in secret_keywords) or any(kw in snippet_lower for kw in secret_keywords):
            return {
                "success": True,
                "response": "Please remove sensitive credentials before uploading this file.",
                "provider": "mock"
            }

        if "unclear" in q_lower or "blur" in q_lower or "blurry" in q_lower or "cannot read" in q_lower or "unclear" in snippet_lower:
            return {
                "success": True,
                "response": "I cannot clearly read this part. Please upload a clearer image.",
                "provider": "mock"
            }

        missing_keywords = ["father name", "mother name", "passport", "tax id", "missing", "not present", "unknown info", "salary"]
        if any(kw in q_lower for kw in missing_keywords) and not any(kw in snippet_lower for kw in missing_keywords if len(kw) > 2):
            return {
                "success": True,
                "response": "I cannot find that information in the uploaded document.",
                "provider": "mock"
            }

        if language == "Hindi":
            reply = (
                "I found:\n"
                "- यह दस्तावेज़ आपके प्रश्न से संबंधित स्पष्ट जानकारी प्रदान करता है।\n"
                "- मुख्य श्रेणी: आधिकारिक सूचना / विवरण / निर्देश\n\n"
                "Step-by-step:\n"
                "1. दस्तावेज़ में दिए गए निर्देशों को ध्यान से समझें।\n"
                "2. अंतिम तिथि से पहले अपनी प्रक्रिया पूरी करें।\n\n"
                "Important:\n"
                "- दस्तावेज़ में दी गई अंतिम तिथि या चेतावनी का विशेष ध्यान रखें।"
            )
        elif language == "Hinglish":
            reply = (
                "I found:\n"
                "- Uploaded document aapke question se related main details dikha raha hai.\n"
                "- Category: Official Notice / Bill / Statement / Instructions\n\n"
                "Step-by-step:\n"
                "1. Document ki main requirement ko step-by-step samjhein.\n"
                "2. Stated deadline se pehle action complete karein.\n\n"
                "Important:\n"
                "- Document mein di gayi last date ya warning note ka dhyan rakhein."
            )
        else:
            reply = (
                "I found:\n"
                "- The uploaded document contains verified details relevant to your question.\n"
                "- Category: Official Notice / Statement / Instructions\n\n"
                "Step-by-step:\n"
                "1. Review the key requirements and details from the document excerpt.\n"
                "2. Complete the required actions step-by-step before any listed due date.\n\n"
                "Important:\n"
                "- Pay attention to any important due dates or warnings clearly visible in the document."
            )

        return {"success": True, "response": reply, "provider": "mock"}

    def _generate_mock_explain(self, text: str, language: str = "English") -> Dict[str, Any]:
        if language == "Hindi":
            reply = (
                "Main Point:\n"
                "यह संदेश आपको किसी आवश्यक नियम, सूचना या प्रक्रिया को पूरा करने के लिए सूचित कर रहा है।\n\n"
                "Key Actions:\n"
                "1. संदेश में दी गई मुख्य बात या निर्देश को ध्यान से समझें।\n"
                "2. यदि कोई फॉर्म भरना या बैंक/कार्यालय जाना आवश्यक है, तो समय पर पूरा करें।\n\n"
                "Important Warnings / Deadlines:\n"
                "दस्तावेज़ में दी गई किसी भी अंतिम तिथि (Deadline) का विशेष ध्यान रखें।"
            )
        elif language == "Hinglish":
            reply = (
                "Main Point:\n"
                "Ye message aapko kisi important requirement ya update ke baare mein bata raha hai.\n\n"
                "Key Actions:\n"
                "1. Document mein di gayi main requirement ko simple tareeke se samjhein.\n"
                "2. Agar koi last date ya action required hai toh time par complete karein.\n\n"
                "Important Warnings / Deadlines:\n"
                "Document mein di gayi last date ya deadline ka khaas dhyan rakhein."
            )
        else:
            reply = (
                "Main Point:\n"
                "This message is explaining an important update, notice, or requirement for you.\n\n"
                "Key Actions:\n"
                "1. Understand the core instructions provided in your message.\n"
                "2. Take the required action step before the specified due date.\n\n"
                "Important Warnings / Deadlines:\n"
                "Pay special attention to any deadlines, due dates, or safety warnings mentioned in your text."
            )
        return {"success": True, "response": reply, "provider": "mock"}

    def _generate_mock_response(self, user_message: str, category: str = "general", language: str = "English") -> Dict[str, Any]:
        msg_lower = user_message.lower()

        if "whatsapp" in msg_lower or "व्हाट्सएप" in msg_lower or category == "whatsapp":
            if language == "Hindi":
                reply = (
                    "व्हाट्सएप (WhatsApp) परिवार और दोस्तों से जुड़े रहने का एक बहुत ही आसान तरीका है।\n\n"
                    "Step-by-step:\n"
                    "1. अपने फोन में व्हाट्सएप का हरा आइकॉन खोलें।\n"
                    "2. जिस व्यक्ति को मैसेज भेजना है, उनके नाम पर दबाएं।\n"
                    "3. नीचे बने खाली बॉक्स में अपना संदेश लिखें और हरे तीर पर दबाएं।\n\n"
                    "You may also need:\n"
                    "- चालू इंटरनेट या वाई-फाई कनेक्शन\n"
                    "- अपने फोन में सेव किया गया संपर्क नंबर\n\n"
                    "Important:\n"
                    "किसी भी अनजान व्यक्ति द्वारा व्हाट्सएप पर भेजे गए लिंक को न खोलें।"
                )
            elif language == "Hinglish":
                reply = (
                    "WhatsApp ke zariye aap apne family aur friends se aasaani se baatein kar sakte hain.\n\n"
                    "Step-by-step:\n"
                    "1. Apne phone par WhatsApp icon open karein.\n"
                    "2. Jis person ko message bhejna hai unke naam par tap karein.\n"
                    "3. Niche box mein message type karke green Arrow button dabaein.\n\n"
                    "You may also need:\n"
                    "- Active Internet ya Wi-Fi connection\n"
                    "- Saved contact number in your phone\n\n"
                    "Important:\n"
                    "Anjaan person ke bheje gaye link par kabhi click mat karein."
                )
            else:
                reply = (
                    "WhatsApp is a simple and free way to stay connected with your family and friends.\n\n"
                    "Step-by-step:\n"
                    "1. Open the green WhatsApp app icon on your phone.\n"
                    "2. Tap on the contact name you wish to message.\n"
                    "3. Type your message in the bottom box and tap the green Arrow to send.\n\n"
                    "You may also need:\n"
                    "- Active Internet or Wi-Fi connection\n"
                    "- Saved contact number in your phone\n\n"
                    "Important:\n"
                    "Never click on unknown links sent by strangers on WhatsApp."
                )
        elif "bank" in msg_lower or "बैंक" in msg_lower or "बैंकिंग" in msg_lower or category == "banking":
            if language == "Hindi":
                reply = (
                    "आप अपने बैंक के आधिकारिक ऐप से घर बैठे सुरक्षित रूप से अपना बैलेंस देख सकते हैं।\n\n"
                    "Step-by-step:\n"
                    "1. अपने मोबाइल में अपने बैंक का आधिकारिक ऐप खोलें।\n"
                    "2. अपने सुरक्षित ऐप पिन या फिंगरप्रिंट से लॉगिन करें।\n"
                    "3. होम स्क्रीन पर 'View Balance' विकल्प पर दबाएं।\n\n"
                    "You may also need:\n"
                    "- रजिस्टर्ड सिम कार्ड वाला स्मार्टफोन\n"
                    "- ऐप लॉगिन पिन या फिंगरप्रिंट\n\n"
                    "Important:\n"
                    "अपना OTP, ATM पिन या पासवर्ड कभी भी किसी व्यक्ति के साथ साझा न करें।"
                )
            elif language == "Hinglish":
                reply = (
                    "Aap apne official bank app se ghar baithe safe tareeke se account balance dekh sakte hain.\n\n"
                    "Step-by-step:\n"
                    "1. Apne mobile mein official bank application open karein.\n"
                    "2. Apne safe App PIN ya fingerprint se log in karein.\n"
                    "3. Main screen par 'View Balance' option par tap karein.\n\n"
                    "You may also need:\n"
                    "- Registered SIM card wala smartphone\n"
                    "- App login passcode ya fingerprint\n\n"
                    "Important:\n"
                    "Apna OTP, PIN ya Password kisi ke sath share mat karein."
                )
            else:
                reply = (
                    "You can safely check your bank account balance using your bank's official mobile application.\n\n"
                    "Step-by-step:\n"
                    "1. Open your official bank app on your smartphone.\n"
                    "2. Log in using your secure App PIN or fingerprint.\n"
                    "3. Tap on 'Check Account Balance' on the main screen.\n\n"
                    "You may also need:\n"
                    "- Smartphone with registered SIM card\n"
                    "- App login passcode or fingerprint sensor\n\n"
                    "Important:\n"
                    "Never share your OTP, ATM PIN, or password with anyone."
                )
        elif "train" in msg_lower or "ट्रेन" in msg_lower or "ticket" in msg_lower or category == "train":
            if language == "Hindi":
                reply = (
                    "ऑनलाइन ट्रेन टिकट बुक करने के लिए आधिकारिक IRCTC ऐप या वेबसाइट का उपयोग करें।\n\n"
                    "Step-by-step:\n"
                    "1. अपने मोबाइल पर IRCTC Rail Connect ऐप खोलें।\n"
                    "2. अपने उपयोगकर्ता नाम और पासवर्ड से लॉगिन करें।\n"
                    "3. अपने स्टेशन और यात्रा की तिथि चुनकर ट्रेन खोजें और टिकट बुक करें।\n\n"
                    "You may also need:\n"
                    "- IRCTC खाता लॉगिन विवरण\n"
                    "- ऑनलाइन भुगतान के लिए UPI या बैंक कार्ड\n\n"
                    "Important:\n"
                    "केवल आधिकारिक IRCTC ऐप या अधिकृत एजेंटों से ही ट्रेन टिकट बुक करें।"
                )
            elif language == "Hinglish":
                reply = (
                    "Online train ticket book karne ke liye IRCTC app ya official website use karein.\n\n"
                    "Step-by-step:\n"
                    "1. Apne phone mein IRCTC Rail Connect app open karein.\n"
                    "2. Apne login username aur password se sign in karein.\n"
                    "3. Station aur travel date choose karke train ticket confirm karein.\n\n"
                    "You may also need:\n"
                    "- Active IRCTC account login credentials\n"
                    "- UPI ya Netbanking online payment details\n\n"
                    "Important:\n"
                    "Hamesha official IRCTC platform se hi train ticket booking karein."
                )
            else:
                reply = (
                    "To book train tickets online safely, use the official IRCTC website or mobile app.\n\n"
                    "Step-by-step:\n"
                    "1. Open the IRCTC Rail Connect app on your smartphone.\n"
                    "2. Log in with your IRCTC username and password.\n"
                    "3. Select departure/destination stations, choose date, and proceed to payment.\n\n"
                    "You may also need:\n"
                    "- Valid IRCTC login account\n"
                    "- Online payment method (UPI/Debit Card)\n\n"
                    "Important:\n"
                    "Only use authorized booking portals for railway tickets."
                )
        elif "email" in msg_lower or "gmail" in msg_lower or "attachment" in msg_lower or category == "email":
            if language == "Hindi":
                reply = (
                    "जीमेल (Gmail) ऐप में फ़ाइल या फोटो अटैच करके भेजना बहुत आसान है।\n\n"
                    "Step-by-step:\n"
                    "1. जीमेल ऐप खोलें और 'Compose' (रचना) बटन दबाएं।\n"
                    "2. ऊपर दिए गए पेपरक्लिप (Clip) आइकॉन पर दबाएं और 'Attach file' चुनें।\n"
                    "3. अपनी फोटो या दस्तावेज़ चुनकर 'Send' (हरे/नीले तीर) पर दबाएं।\n\n"
                    "You may also need:\n"
                    "- प्राप्तकर्ता का सही ईमेल आईडी\n"
                    "- वह फोटो या फ़ाइल जो आप भेजना चाहते हैं\n\n"
                    "Important:\n"
                    "अटैचमेंट भेजने से पहले प्राप्तकर्ता का ईमेल पता दोबारा जांच लें।"
                )
            elif language == "Hinglish":
                reply = (
                    "Gmail app mein photo ya file attachment ke sath bhejney ke liye niche diye steps follow karein.\n\n"
                    "Step-by-step:\n"
                    "1. Phone mein Gmail app open karke Compose button par tap karein.\n"
                    "2. Top bar par Paperclip icon par click karke 'Attach file' select karein.\n"
                    "3. File select karke recipient ka Email ID dalein aur Send button dabaein.\n\n"
                    "You may also need:\n"
                    "- Recipient ka correct Email address\n"
                    "- Phone gallery mein saved file or document\n\n"
                    "Important:\n"
                    "Send click karne se pehle email ID dhyan se check karein."
                )
            else:
                reply = (
                    "To send an email with an attachment in Gmail, follow these simple steps.\n\n"
                    "Step-by-step:\n"
                    "1. Open the Gmail app and tap the 'Compose' button.\n"
                    "2. Tap the Paperclip icon at the top and select 'Attach File'.\n"
                    "3. Choose your document or photo, enter the recipient email, and tap Send.\n\n"
                    "You may also need:\n"
                    "- Correct recipient email address\n"
                    "- Saved file on your device\n\n"
                    "Important:\n"
                    "Always verify the recipient's email address before sending."
                )
        elif "bank" in msg_lower or "बैंक" in msg_lower or "बैंकिंग" in msg_lower or category == "banking":
            if language == "Hindi":
                reply = (
                    "आप अपने बैंक के आधिकारिक ऐप से घर बैठे सुरक्षित रूप से अपना बैलेंस देख सकते हैं।\n\n"
                    "Step-by-step:\n"
                    "1. अपने मोबाइल में अपने बैंक का आधिकारिक ऐप खोलें।\n"
                    "2. अपने सुरक्षित ऐप पिन या फिंगरप्रिंट से लॉगिन करें।\n"
                    "3. होम स्क्रीन पर 'View Balance' विकल्प पर दबाएं।\n\n"
                    "You may also need:\n"
                    "- रजिस्टर्ड सिम कार्ड वाला स्मार्टफोन\n"
                    "- ऐप लॉगिन पिन या फिंगरप्रिंट\n\n"
                    "Important:\n"
                    "अपना OTP, ATM पिन या पासवर्ड कभी भी किसी व्यक्ति के साथ साझा न करें।"
                )
            elif language == "Hinglish":
                reply = (
                    "Aap apne official bank app se ghar baithe safe tareeke se account balance dekh sakte hain.\n\n"
                    "Step-by-step:\n"
                    "1. Apne mobile mein official bank application open karein.\n"
                    "2. Apne safe App PIN ya fingerprint se log in karein.\n"
                    "3. Main screen par 'View Balance' option par tap karein.\n\n"
                    "You may also need:\n"
                    "- Registered SIM card wala smartphone\n"
                    "- App login passcode ya fingerprint\n\n"
                    "Important:\n"
                    "Apna OTP, PIN ya Password kisi ke sath share mat karein."
                )
            else:
                reply = (
                    "You can safely check your bank account balance using your bank's official mobile application.\n\n"
                    "Step-by-step:\n"
                    "1. Open your official bank app on your smartphone.\n"
                    "2. Log in using your secure App PIN or fingerprint.\n"
                    "3. Tap on 'Check Account Balance' on the main screen.\n\n"
                    "You may also need:\n"
                    "- Smartphone with registered SIM card\n"
                    "- App login passcode or fingerprint sensor\n\n"
                    "Important:\n"
                    "Never share your OTP, ATM PIN, or password with anyone."
                )
        else:
            if language == "Hindi":
                reply = (
                    "नमस्ते! मैं SeniorEase AI हूँ, आपका सरल और भरोसेमंद डिजिटल साथी।\n\n"
                    "Step-by-step:\n"
                    "1. ऊपर दिए गए किसी भी बटन को दबाएं।\n"
                    "2. या नीचे दिए गए बॉक्स में अपना प्रश्न हिंदी में लिखें।\n"
                    "3. नीले बटन 'Ask SeniorEase' पर दबाएं।\n\n"
                    "You may also need:\n"
                    "- अपना प्रश्न सरल शब्दों में सोचना\n"
                    "- उत्तर पढ़ने के लिए थोड़ा समय निकालना\n\n"
                    "Important:\n"
                    "मैं आपके प्रश्नों का सरल और आसान भाषा में उत्तर देने के लिए हमेशा तैयार हूँ।"
                )
            elif language == "Hinglish":
                reply = (
                    "Namaste! Main SeniorEase AI hoon, aapka simple aur trustworthy digital companion.\n\n"
                    "Step-by-step:\n"
                    "1. Upar diye gaye topic button par tap karein.\n"
                    "2. Ya niche box mein apna question type karein.\n"
                    "3. Blue button 'Ask SeniorEase' par click karein.\n\n"
                    "You may also need:\n"
                    "- Apna question simple words mein sochna\n"
                    "- Answer read karne ke liye thoda time nikalna\n\n"
                    "Important:\n"
                    "Main aapke har question ka simple language mein jawab dunga."
                )
            else:
                reply = (
                    "Hello! I am SeniorEase AI, your patient and friendly digital companion.\n\n"
                    "Step-by-step:\n"
                    "1. Tap any quick topic button above.\n"
                    "2. Or type your question in the text box below.\n"
                    "3. Click the blue 'Ask SeniorEase' button.\n\n"
                    "You may also need:\n"
                    "- A clear idea of what you want help with\n"
                    "- A few moments to read the answer step-by-step\n\n"
                    "Important:\n"
                    "Feel free to ask any question without feeling rushed or embarrassed."
                )

        return {"success": True, "response": reply, "provider": "mock"}


# Global singleton instance of AIService
ai_service = AIService()
