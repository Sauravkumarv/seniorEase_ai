import os
import re
import logging
from typing import Dict, Any

# Configure logging for service operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Master System Prompt for SeniorEase AI Chat
SENIOR_EASE_SYSTEM_PROMPT = """
ROLE:
You are SeniorEase AI, a patient, friendly and trustworthy digital assistant for senior citizens.

COMMUNICATION:
- Use simple language.
- Avoid technical jargon.
- Explain unfamiliar terms simply when they appear.
- Keep sentences short and clear.
- Give instructions step-by-step.
- Use numbered lists for all instructions.
- Be patient and reassuring.
- Never make the user feel embarrassed for asking a basic question.

LANGUAGE INSTRUCTIONS:
- You support: English, Hindi, Hinglish.
- If the user selects Hindi, respond primarily in Hindi (using Devanagari script).
- If the user selects Hinglish, respond in simple Hindi-English (using English/Roman script).
- If English is selected, respond in simple English.

IMPORTANT SAFETY:
- Never ask the user to provide: OTP, ATM PIN, UPI PIN, CVV, Password, Full card number, or Banking credentials.
- If a user shares sensitive information (like an OTP, PIN, password, or card number), immediately tell them NOT to share it and recommend removing it from the conversation.
- For financial, medical, legal, or government-related topics:
  * Provide general guidance only.
  * Clearly mention when the user should verify information from the official source (such as their bank branch, doctor, or official government portal).
  * Do not pretend to be an official representative.

RESPONSE FORMAT:
Start with a short direct answer.

Then provide:

Step-by-step:
1. Step one
2. Step two
3. Step three

If relevant to help the user prepare, optionally include:

You may also need:
- First requirement or item
- Second requirement or item
- Third requirement or item

If useful, add:

Important:
A short warning or useful tip.

Keep responses concise, reassuring, and easy to read.
"""

# System Prompt for "Explain Something Simply" feature
EXPLAIN_SYSTEM_PROMPT = """
ROLE:
You are SeniorEase AI, a patient and helpful digital companion for senior citizens. Your job is to simplify complex text, bank notices, government letters, legal jargon, or difficult instructions.

RULES:
1. Remove unnecessary technical, legal, or bureaucratic jargon.
2. Explain any difficult or complex terms in simple everyday language.
3. Summarize the main point clearly in 1-2 short sentences.
4. List important action steps using a clear numbered list.
5. PRESERVE all critical warnings, safety notes, due dates, or deadlines.
6. Use simple, warm, and easy-to-read language.

LANGUAGE INSTRUCTIONS:
- If language is Hindi: respond in simple Hindi using Devanagari script.
- If language is Hinglish: respond in simple Hinglish (Hindi written in Roman/English script).
- If language is English: respond in simple English.

RESPONSE FORMAT:
Main Point:
[1-2 clear, simple sentences summarizing what the message means]

Key Actions:
1. First action step
2. Second action step

Important Warnings / Deadlines:
[Preserved deadlines, due dates, or safety warnings if mentioned in the text]
"""

class AIService:
    """
    AIService manages interactions with AI providers (Gemini API & OpenAI API)
    and provides a built-in mock fallback with proactive assistance and privacy protection.
    """

    def __init__(self):
        self.provider = os.getenv("AI_PROVIDER", "mock").lower()
        self.api_key = os.getenv("AI_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("AI_MODEL", "gemini-1.5-flash")

        logger.info(f"Initialized AIService with provider: '{self.provider}' and model: '{self.model}'")

    def generate_response(self, user_message: str, category: str = "general", language: str = "English") -> Dict[str, Any]:
        """
        Generates a senior-friendly response adhering strictly to SeniorEase AI guidelines.
        """
        if not user_message or not user_message.strip():
            return {
                "success": False,
                "error": "Message cannot be empty."
            }

        cleaned_message = user_message.strip()

        # Intercept sensitive credentials locally before calling AI providers
        sensitive_warning = self._check_sensitive_information(cleaned_message, language)
        if sensitive_warning:
            return {
                "success": True,
                "response": sensitive_warning,
                "provider": "safety_filter"
            }

        # Check if mock fallback mode is active or API key is unconfigured
        is_mock_key = not self.api_key or self.api_key.lower() in ["mock", "your_api_key_here", "none"]
        if self.provider == "mock" or is_mock_key:
            return self._generate_mock_response(cleaned_message, category, language)

        # Handle Gemini API provider
        if self.provider in ["gemini", "google"]:
            return self._call_gemini_api(cleaned_message, language)

        # Handle OpenAI API provider
        if self.provider == "openai":
            return self._call_openai_api(cleaned_message, language)

        logger.warning(f"Unrecognized provider '{self.provider}'. Falling back to mock service.")
        return self._generate_mock_response(cleaned_message, category, language)

    def explain_text(self, text: str, language: str = "English") -> Dict[str, Any]:
        """
        Simplifies complex messages, notices, or difficult text for seniors.
        """
        if not text or not text.strip():
            return {
                "success": False,
                "error": "Text to explain cannot be empty."
            }

        cleaned_text = text.strip()

        # Intercept sensitive credentials locally before calling AI providers
        sensitive_warning = self._check_sensitive_information(cleaned_text, language)
        if sensitive_warning:
            return {
                "success": True,
                "response": sensitive_warning,
                "provider": "safety_filter"
            }

        is_mock_key = not self.api_key or self.api_key.lower() in ["mock", "your_api_key_here", "none"]
        if self.provider == "mock" or is_mock_key:
            return self._generate_mock_explain(cleaned_text, language)

        if self.provider in ["gemini", "google"]:
            return self._call_gemini_explain(cleaned_text, language)

        if self.provider == "openai":
            return self._call_openai_explain(cleaned_text, language)

        return self._generate_mock_explain(cleaned_text, language)

    def _check_sensitive_information(self, text: str, language: str) -> str:
        """
        Intercepts sensitive credentials (OTP, PIN, CVV, Password) locally.
        Ensures sensitive text is NEVER logged and NEVER sent to external AI providers.
        """
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

    def _call_gemini_api(self, user_message: str, language: str = "English") -> Dict[str, Any]:
        """
        Executes request via Google Gemini API using REST endpoint.
        """
        try:
            import requests

            lang_directive = f"\nUSER LANGUAGE PREFERENCE: {language}."
            if language == "Hindi":
                lang_directive += " Respond primarily in simple Hindi using Devanagari script."
            elif language == "Hinglish":
                lang_directive += " Respond in simple Hinglish (Hindi words written using English/Roman script)."
            else:
                lang_directive += " Respond in simple English."

            full_system_prompt = SENIOR_EASE_SYSTEM_PROMPT + lang_directive
            model_name = self.model if "gemini" in self.model else "gemini-1.5-flash"
            gemini_key = self.api_key or os.getenv("GEMINI_API_KEY", "")

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": full_system_prompt + "\n\nUser Question:\n" + user_message}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.5,
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
                            "provider": f"gemini ({model_name})"
                        }

            logger.error(f"Gemini API error {res.status_code}: {res.text}")
            mock_res = self._generate_mock_response(user_message, language=language)
            mock_res["warning"] = f"Gemini API response issue (Code {res.status_code}). Showing demonstration response."
            return mock_res

        except Exception as e:
            logger.error(f"Gemini API call exception: {e}")
            mock_res = self._generate_mock_response(user_message, language=language)
            mock_res["warning"] = f"Gemini API connection error ({str(e)}). Showing demonstration response."
            return mock_res

    def _call_gemini_explain(self, text: str, language: str = "English") -> Dict[str, Any]:
        """
        Executes text simplification via Google Gemini API REST endpoint.
        """
        try:
            import requests

            lang_directive = f"\nUSER LANGUAGE PREFERENCE: {language}."
            if language == "Hindi":
                lang_directive += " Respond in simple Hindi using Devanagari script."
            elif language == "Hinglish":
                lang_directive += " Respond in simple Hinglish (Hindi written in Roman script)."
            else:
                lang_directive += " Respond in simple English."

            full_system_prompt = EXPLAIN_SYSTEM_PROMPT + lang_directive
            model_name = self.model if "gemini" in self.model else "gemini-1.5-flash"
            gemini_key = self.api_key or os.getenv("GEMINI_API_KEY", "")

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": full_system_prompt + "\n\nPlease simplify this message for me:\n\n" + text}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.5,
                    "maxOutputTokens": 600
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
                            "provider": f"gemini ({model_name})"
                        }

            logger.error(f"Gemini API explain error {res.status_code}: {res.text}")
            mock_res = self._generate_mock_explain(text, language=language)
            mock_res["warning"] = f"Gemini API response issue (Code {res.status_code}). Showing demonstration response."
            return mock_res

        except Exception as e:
            logger.error(f"Gemini API explain exception: {e}")
            mock_res = self._generate_mock_explain(text, language=language)
            mock_res["warning"] = f"Gemini API connection error ({str(e)}). Showing demonstration response."
            return mock_res

    def _call_openai_api(self, user_message: str, language: str = "English") -> Dict[str, Any]:
        """
        Executes request via OpenAI API using the SeniorEase AI System Prompt.
        """
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

            full_system_prompt = SENIOR_EASE_SYSTEM_PROMPT + lang_directive

            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.6,
                max_tokens=650
            )

            response_text = completion.choices[0].message.content
            return {
                "success": True,
                "response": response_text,
                "provider": "openai"
            }

        except Exception as e:
            logger.error(f"OpenAI API call error: {e}")
            mock_res = self._generate_mock_response(user_message, language=language)
            mock_res["warning"] = f"API connection issue ({str(e)}). Showing offline demonstration response."
            return mock_res

    def _call_openai_explain(self, text: str, language: str = "English") -> Dict[str, Any]:
        """
        Calls OpenAI API specifically to simplify difficult text for seniors.
        """
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

            full_system_prompt = EXPLAIN_SYSTEM_PROMPT + lang_directive

            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": f"Please simplify this message for me:\n\n{text}"}
                ],
                temperature=0.5,
                max_tokens=600
            )

            response_text = completion.choices[0].message.content
            return {
                "success": True,
                "response": response_text,
                "provider": "openai"
            }

        except Exception as e:
            logger.error(f"OpenAI API explain error: {e}")
            mock_res = self._generate_mock_explain(text, language=language)
            mock_res["warning"] = f"API connection issue ({str(e)}). Showing offline demonstration response."
            return mock_res

    def _generate_mock_explain(self, text: str, language: str = "English") -> Dict[str, Any]:
        """
        Mock generator for text simplification in English, Hindi, and Hinglish.
        """
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

        return {
            "success": True,
            "response": reply,
            "provider": "mock"
        }

    def _generate_mock_response(self, user_message: str, category: str = "general", language: str = "English") -> Dict[str, Any]:
        """
        Generates structured responses adhering strictly to SeniorEase AI guidelines including proactive assistance.
        """
        msg_lower = user_message.lower()

        # WhatsApp Help Topic
        if "whatsapp" in msg_lower or category == "whatsapp":
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

        # Banking Help Topic
        elif "bank" in msg_lower or category == "banking":
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

        # Train Booking Help Topic
        elif "train" in msg_lower or "irctc" in msg_lower or category == "train":
            if language == "Hindi":
                reply = (
                    "आधिकारिक IRCTC ऐप या वेबसाइट से ट्रेन का टिकट ऑनलाइन बुक करना बहुत आसान है।\n\n"
                    "Step-by-step:\n"
                    "1. रेलवे बुकिंग वेबसाइट/ऐप खोलें।\n"
                    "2. अपनी यात्रा का विवरण दर्ज करें।\n"
                    "3. अपनी ट्रेन चुनें।\n"
                    "4. यात्री विवरण की समीक्षा करें।\n"
                    "5. भुगतान पूरा करें।\n\n"
                    "You may also need:\n"
                    "- आपकी यात्रा की तारीख और स्टेशन का नाम\n"
                    "- यात्री का फोटो आईडी विवरण\n"
                    "- भुगतान का तरीका (UPI/कार्ड/नेट बैंकिंग)\n\n"
                    "Important:\n"
                    "अपना OTP या PIN किसी के साथ शेयर न करें।"
                )
            elif language == "Hinglish":
                reply = (
                    "Official IRCTC app ya website se train ticket book karna bahut aasan hai.\n\n"
                    "Step-by-step:\n"
                    "1. Railway booking website/app open karein.\n"
                    "2. Apni journey details enter karein.\n"
                    "3. Apni train select karein.\n"
                    "4. Passenger details review karein.\n"
                    "5. Payment complete karein.\n\n"
                    "You may also need:\n"
                    "- Aapki travel date aur station names\n"
                    "- Passenger Photo ID details\n"
                    "- Payment method (UPI/Net banking/Card)\n\n"
                    "Important:\n"
                    "Apna OTP ya PIN kisi ke sath share mat karein."
                )
            else:
                reply = (
                    "Booking a train ticket online is simple using the official IRCTC app or portal.\n\n"
                    "Step-by-step:\n"
                    "1. Open the railway booking website/app.\n"
                    "2. Enter your journey details.\n"
                    "3. Select your train.\n"
                    "4. Review passenger details.\n"
                    "5. Complete payment.\n\n"
                    "You may also need:\n"
                    "- Your journey date\n"
                    "- Passenger ID details\n"
                    "- Payment method\n\n"
                    "Important:\n"
                    "Never share your OTP or PIN with anyone."
                )

        # Email Help Topic
        elif "email" in msg_lower or category == "email":
            if language == "Hindi":
                reply = (
                    "जीमेल (Gmail) के जरिए ईमेल भेजना बहुत ही सरल है।\n\n"
                    "Step-by-step:\n"
                    "1. अपने फोन में जीमेल (Gmail) ऐप खोलें।\n"
                    "2. नीचे दिए गए 'Compose' (प्लस ➕) बटन पर दबाएं।\n"
                    "3. 'To' में प्राप्तकर्ता की ईमेल आईडी लिखें।\n"
                    "4. अपना संदेश लिखें।\n"
                    "5. 'Send' (नीला तीर) पर दबाएं।\n\n"
                    "You may also need:\n"
                    "- प्राप्तकर्ता की सही ईमेल आईडी\n"
                    "- चालू इंटरनेट कनेक्शन\n\n"
                    "Important:\n"
                    "किसी भी अनजान ईमेल में आई फाइलों या लिंक को न खोलें।"
                )
            elif language == "Hinglish":
                reply = (
                    "Gmail se email bhejna bahut simple aur safe hai.\n\n"
                    "Step-by-step:\n"
                    "1. Apne phone mein Gmail app open karein.\n"
                    "2. Niche bane Compose (plus ➕) button par tap karein.\n"
                    "3. 'To' field mein receiver ki email ID likhein.\n"
                    "4. Apna message type karein.\n"
                    "5. Send arrow button par tap karein.\n\n"
                    "You may also need:\n"
                    "- Receiver ki exact email ID\n"
                    "- Active Internet connection\n\n"
                    "Important:\n"
                    "Anjaan email ke kisi bhi link ya file ko open mat karein."
                )
            else:
                reply = (
                    "Sending an email on Gmail takes just a few simple steps.\n\n"
                    "Step-by-step:\n"
                    "1. Open the Gmail app on your mobile phone.\n"
                    "2. Tap the Compose button (marked with a ➕ plus icon).\n"
                    "3. Type the recipient's email address in the 'To' field.\n"
                    "4. Write your message.\n"
                    "5. Tap the Send arrow button.\n\n"
                    "You may also need:\n"
                    "- Recipient's exact email address\n"
                    "- Internet connection\n\n"
                    "Important:\n"
                    "Do not open attachments or click links from unknown senders."
                )

        # Online Shopping Help Topic
        elif "shop" in msg_lower or category == "shopping":
            if language == "Hindi":
                reply = (
                    "ऑनलाइन शॉपिंग से आप घर बैठे सामान आसानी से मंगवा सकते हैं।\n\n"
                    "Step-by-step:\n"
                    "1. भरोसेमंद ऐप जैसे Amazon या Flipkart खोलें।\n"
                    "2. सर्च बार में सामान का नाम लिखें।\n"
                    "3. सामान चुनें और 'Add to Cart' पर दबाएं।\n"
                    "4. चेकआउट प्रक्रिया पर जाएं।\n"
                    "5. 'Cash on Delivery' विकल्प चुनें।\n\n"
                    "You may also need:\n"
                    "- घर का डिलीवरी पता\n"
                    "- डिलीवरी अपडेट के लिए मोबाइल नंबर\n\n"
                    "Important:\n"
                    "'Cash on Delivery' चुनने से आप सामान हाथ में मिलने के बाद ही पैसे देते हैं।"
                )
            elif language == "Hinglish":
                reply = (
                    "Online shopping se aap ghar baithe saman mangwa sakte hain.\n\n"
                    "Step-by-step:\n"
                    "1. Trusted shopping app jaise Amazon ya Flipkart open karein.\n"
                    "2. Search bar mein item search karke product select karein.\n"
                    "3. 'Add to Cart' par tap karein.\n"
                    "4. Checkout page par jayein.\n"
                    "5. 'Cash on Delivery' choose karein.\n\n"
                    "You may also need:\n"
                    "- Full delivery address details\n"
                    "- Mobile number for delivery updates\n\n"
                    "Important:\n"
                    "'Cash on Delivery' choose karne se saman ghar aane ke baad hi payment karna hota hai."
                )
            else:
                reply = (
                    "Shopping online allows you to order products safely to your home.\n\n"
                    "Step-by-step:\n"
                    "1. Open a trusted shopping app like Amazon or Flipkart.\n"
                    "2. Type the item name in the search bar and select your product.\n"
                    "3. Tap 'Add to Cart' and proceed to checkout.\n"
                    "4. Select 'Cash on Delivery'.\n\n"
                    "You may also need:\n"
                    "- Delivery address details\n"
                    "- Mobile number for delivery updates\n\n"
                    "Important:\n"
                    "Choosing Cash on Delivery allows you to inspect your order and pay safely when it arrives."
                )

        # Government & Pension Services Help Topic
        elif "gov" in msg_lower or "pension" in msg_lower or "aadhar" in msg_lower or category == "gov":
            if language == "Hindi":
                reply = (
                    "आप घर बैठे आसानी से अपना डिजिटल जीवन प्रमाण पत्र (Pension Life Certificate) जमा कर सकते हैं।\n\n"
                    "Step-by-step:\n"
                    "1. उमंग (UMANG) ऐप डाउनलोड करें या डाकिया (Postman) से संपर्क करें।\n"
                    "2. अपना आधार नंबर और पेंशन PPO विवरण दर्ज करें।\n"
                    "3. चेहरा (Face Verification) दिखाकर डिजिटल जीवन प्रमाण पत्र जमा करें।\n\n"
                    "You may also need:\n"
                    "- पेंशन PPO नंबर\n"
                    "- बैंक खाते से जुड़ा आधार नंबर\n"
                    "- फ्रंट कैमरे वाला स्मार्टफोन\n\n"
                    "Important:\n"
                    "सरकारी योजनाओं की जानकारी हमेशा आधिकारिक '.gov.in' वेबसाइट से ही सत्यापित करें।"
                )
            elif language == "Hinglish":
                reply = (
                    "Aap ghar baithe aasaani se apna Life Certificate submit kar sakte hain.\n\n"
                    "Step-by-step:\n"
                    "1. UMANG app download karein ya Postman doorstep service ki help lein.\n"
                    "2. Apna Aadhaar number aur Pension PPO details enter karein.\n"
                    "3. Face authentication se digital Life Certificate submit karein.\n\n"
                    "You may also need:\n"
                    "- Pension PPO Number\n"
                    "- Aadhaar number linked with pension\n"
                    "- Smartphone with front camera\n\n"
                    "Important:\n"
                    "Government services ki jankari ke liye hamesha official '.gov.in' portals hi verify karein."
                )
            else:
                reply = (
                    "You can conveniently submit your Pension Life Certificate (Jeevan Pramaan) from home.\n\n"
                    "Step-by-step:\n"
                    "1. Download the official UMANG app or contact India Post Doorstep Banking.\n"
                    "2. Enter your Aadhaar number and Pension PPO details.\n"
                    "3. Complete face verification to submit your digital life certificate.\n\n"
                    "You may also need:\n"
                    "- Pension PPO Number\n"
                    "- Aadhaar number linked to pension account\n"
                    "- Smartphone with front camera\n\n"
                    "Important:\n"
                    "Always verify government scheme details on official '.gov.in' portals or visit your local pension office."
                )

        # General Default Response
        else:
            if language == "Hindi":
                reply = (
                    "नमस्ते! मैं SeniorEase AI हूँ, आपका सरल और भरोसेमंद डिजिटल साथी।\n\n"
                    "Step-by-step:\n"
                    "1. ऊपर दिए गए किसी भी बटन (जैसे व्हाट्सएप या बैंकिंग) को दबाएं।\n"
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
                    "1. Upar दिए गए किसी भी topic button par tap karein.\n"
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
                    "1. Tap any quick topic button above (like WhatsApp or Banking).\n"
                    "2. Or type your question in the text box below.\n"
                    "3. Click the blue 'Ask SeniorEase' button.\n\n"
                    "You may also need:\n"
                    "- A clear idea of what you want help with\n"
                    "- A few moments to read the answer step-by-step\n\n"
                    "Important:\n"
                    "Feel free to ask any question without feeling rushed or embarrassed."
                )

        return {
            "success": True,
            "response": reply,
            "provider": "mock"
        }

# Global singleton instance of AIService
ai_service = AIService()
