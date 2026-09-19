from flask import Blueprint, request, jsonify
from datetime import datetime
import sys
import os

# Add parent directory to sys.path to allow clean imports across modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.services.ai_service import ai_service

# Define Flask Blueprint for Chat API routes
chat_bp = Blueprint('chat', __name__, url_prefix='/api')

@chat_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify backend service status.
    Used by Streamlit frontend to confirm API connection.
    """
    return jsonify({
        "status": "healthy",
        "service": "SeniorEase AI Backend REST API",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }), 200

@chat_bp.route('/chat', methods=['POST'])
def chat():
    """
    Main Chat API Endpoint.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "Invalid JSON request payload."
            }), 400

        user_message = data.get("message", "").strip()
        category = data.get("category", "general")
        language = data.get("language", "English")

        if not user_message:
            return jsonify({
                "status": "error",
                "message": "The 'message' field is required."
            }), 400

        result = ai_service.generate_response(
            user_message=user_message,
            category=category,
            language=language
        )

        if not result.get("success"):
            return jsonify({
                "status": "error",
                "message": result.get("error", "Failed to generate AI response.")
            }), 500

        response_payload = {
            "status": "success",
            "response": result.get("response"),
            "provider": result.get("provider", "unknown"),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        if "warning" in result:
            response_payload["warning"] = result["warning"]

        return jsonify(response_payload), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500

@chat_bp.route('/explain', methods=['POST'])
def explain():
    """
    Explain Endpoint for simplifying difficult messages, notices, or instructions.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "Invalid JSON request payload."
            }), 400

        text = data.get("text", "").strip()
        language = data.get("language", "English")

        if not text:
            return jsonify({
                "success": False,
                "message": "The 'text' field is required."
            }), 400

        result = ai_service.explain_text(text=text, language=language)

        if not result.get("success"):
            return jsonify({
                "success": False,
                "message": result.get("error", "Failed to simplify text.")
            }), 500

        return jsonify({
            "success": True,
            "response": result.get("response")
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500

@chat_bp.route('/analyze-image', methods=['POST'])
def analyze_image():
    """
    Image Analysis Endpoint for senior citizens to analyze medicine labels, bills, notices.
    Expects JSON payload:
    {
        "image_base64": "base64 string",
        "mime_type": "image/jpeg",
        "prompt": "optional question",
        "language": "English | Hindi | Hinglish"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "Invalid JSON request payload."
            }), 400

        image_b64 = data.get("image_base64", "").strip()
        mime_type = data.get("mime_type", "image/jpeg")
        prompt = data.get("prompt", "").strip()
        language = data.get("language", "English")

        if not image_b64:
            return jsonify({
                "success": False,
                "message": "The 'image_base64' field is required."
            }), 400

        result = ai_service.analyze_image(
            image_b64=image_b64,
            mime_type=mime_type,
            prompt=prompt,
            language=language
        )

        if not result.get("success"):
            return jsonify({
                "success": False,
                "message": result.get("error", "Failed to analyze image.")
            }), 500

        return jsonify({
            "success": True,
            "response": result.get("response"),
            "provider": result.get("provider", "unknown")
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500

@chat_bp.route('/analyze-doc', methods=['POST'])
def analyze_doc():
    """
    Document Analysis Endpoint for PDF/TXT files.
    Expects JSON payload:
    {
        "document_text": "Extracted text content",
        "question": "Optional user question",
        "language": "English | Hindi | Hinglish"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "Invalid JSON request payload."
            }), 400

        document_text = data.get("document_text", "").strip()
        question = data.get("question", "").strip()
        language = data.get("language", "English")

        if not document_text:
            return jsonify({
                "success": False,
                "message": "The 'document_text' field is required."
            }), 400

        result = ai_service.analyze_document(
            document_text=document_text,
            question=question,
            language=language
        )

        if not result.get("success"):
            return jsonify({
                "success": False,
                "message": result.get("error", "Failed to analyze document.")
            }), 500

        return jsonify({
            "success": True,
            "response": result.get("response"),
            "provider": result.get("provider", "unknown")
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500

@chat_bp.route('/tts', methods=['POST'])
def tts():
    """
    Text-to-Speech Endpoint generating spoken audio for senior users.
    Expects JSON payload:
    {
        "text": "Text to read aloud",
        "language": "English | Hindi | Hinglish"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "Invalid JSON request payload."
            }), 400

        text = data.get("text", "").strip()
        language = data.get("language", "English")

        if not text:
            return jsonify({
                "success": False,
                "message": "The 'text' field is required."
            }), 400

        result = ai_service.generate_tts(text=text, language=language)

        if not result.get("success"):
            return jsonify({
                "success": False,
                "message": result.get("error", "Failed to generate audio.")
            }), 500

        return jsonify({
            "success": True,
            "audio_b64": result.get("audio_b64"),
            "mime_type": result.get("mime_type", "audio/mp3")
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500

@chat_bp.route('/voice/transcribe', methods=['POST'])
def voice_transcribe():
    """
    Voice Transcription Endpoint.
    Converts user speech audio to text using configured Speech-to-Text service.
    Accepts audio file upload via multipart/form-data or JSON payload with base64 audio.
    """
    try:
        audio_bytes = None
        filename = "audio.wav"
        language = "English"

        # 1. Check multipart form file upload
        if 'file' in request.files:
            file_obj = request.files['file']
            audio_bytes = file_obj.read()
            filename = file_obj.filename or "audio.wav"
            language = request.form.get("language", "English")
        elif 'audio' in request.files:
            file_obj = request.files['audio']
            audio_bytes = file_obj.read()
            filename = file_obj.filename or "audio.wav"
            language = request.form.get("language", "English")
        # 2. Check JSON payload with base64 audio string
        elif request.is_json and request.get_json():
            data = request.get_json()
            audio_b64 = data.get("audio_base64", "").strip() or data.get("audio_b64", "").strip()
            if audio_b64:
                import base64
                audio_bytes = base64.b64decode(audio_b64)
            filename = data.get("filename", "audio.wav")
            language = data.get("language", "English")

        if not audio_bytes:
            return jsonify({
                "success": False,
                "message": "No audio file or data provided. Please record your voice and try again."
            }), 400

        result = ai_service.transcribe_audio(
            audio_bytes=audio_bytes,
            filename=filename,
            language=language
        )

        if not result.get("success"):
            return jsonify({
                "success": False,
                "message": result.get("error", "Unable to transcribe audio. Please try speaking clearly or typing your question.")
            }), 500

        return jsonify({
            "success": True,
            "text": result.get("text", "")
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": "Unable to transcribe voice recording. Please try speaking clearly or typing your question."
        }), 500
