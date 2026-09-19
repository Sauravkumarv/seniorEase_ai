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
    Expects JSON payload:
    {
        "message": "User question or prompt",
        "category": "optional category string (whatsapp, banking, train, etc.)",
        "language": "English | Hindi | Hinglish"
    }
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

        # Delegate request processing to the AI service
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
    Expects JSON payload:
    {
        "text": "Difficult message text to explain",
        "language": "Hindi | English | Hinglish"
    }
    Returns JSON response:
    {
        "success": true,
        "response": "Simplified explanation text"
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
