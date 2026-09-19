import os
import sys
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from .env file
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path)

from backend.routes.chat import chat_bp

def create_app() -> Flask:
    """
    Application factory for initializing the Flask REST API backend.
    """
    app = Flask(__name__)

    # Allow the Streamlit Cloud frontend (and local UI) to call this API
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Register blueprints
    app.register_blueprint(chat_bp)

    @app.route('/', methods=['GET'])
    def index():
        return jsonify({
            "name": "SeniorEase AI Backend REST API",
            "version": "1.1.0",
            "endpoints": {
                "health": "/api/health",
                "chat": "/api/chat (POST)",
                "explain": "/api/explain (POST)",
                "analyze_image": "/api/analyze-image (POST)",
                "analyze_doc": "/api/analyze-doc (POST)",
                "tts": "/api/tts (POST)"
            },
            "status": "running"
        }), 200

    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            "status": "error",
            "message": "Resource not found."
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "status": "error",
            "message": "Internal server error."
        }), 500

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.getenv('PORT', os.getenv('FLASK_PORT', 5000)))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    print(f"==================================================")
    print(f"[SeniorEase AI] Backend running on http://127.0.0.1:{port}")
    print(f"==================================================")

    app.run(host='0.0.0.0', port=port, debug=debug)
