# SeniorEase AI 👵👴

**SeniorEase AI** is a simple, accessible, and friendly AI companion designed specifically for senior citizens who may find modern digital tools complex or confusing.

---

## 🌟 Key Features

- **Senior-Friendly Frontend**: Built with Streamlit, featuring large fonts, high contrast colors, simple navigation, and touch-friendly buttons.
- **RESTful Flask Backend**: Runs on port `5000` with modular routing, input validation, and clear JSON API responses.
- **Patient & Clear AI Assistant**: System prompts and responses tailored to deliver empathetic, step-by-step guidance without complex jargon.
- **Smart Mock & Real API Mode**: Operates out-of-the-box in local mock mode without requiring an active paid API key, while fully supporting real OpenAI API integration when configured.
- **Quick Assistance Topics**: One-click quick questions for smartphone help, health term explanations, message drafting, and tech definitions.

---

## 🛠 Tech Stack

- **Python**: Core programming language
- **Flask**: Backend REST API (`port 5000`)
- **Streamlit**: Frontend User Interface (`port 8501`)
- **requests**: Frontend-to-Backend HTTP communication
- **python-dotenv**: Environment variable management
- **openai**: Optional AI service integration

---

## 📁 Project Structure

```text
senior-ease-ai/
│
├── frontend/
│   └── app.py                # Streamlit user interface
│
├── backend/
│   ├── app.py                # Flask REST API entry point (port 5000)
│   ├── routes/
│   │   └── chat.py           # Chat & health API routes
│   └── services/
│       └── ai_service.py     # AI service layer with mock fallback
│
├── .env                      # Environment configuration file
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Directory

Ensure you are in the project folder:
```bash
cd senior-ease-ai
```

### 2. Install Dependencies

It is recommended to use a virtual environment:
```bash
# Create virtual environment (optional but recommended)
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Default contents of `.env`:
```env
FLASK_PORT=5000
FLASK_DEBUG=True
STREAMLIT_PORT=8501
BACKEND_URL=http://127.0.0.1:5000
AI_PROVIDER=mock
AI_API_KEY=mock
AI_MODEL=gpt-3.5-turbo
```

*(Note: If `AI_PROVIDER` is set to `mock`, the app runs completely offline in demonstration mode without requiring an OpenAI key).*

---

## 🏃 Running the Application

### Step 1: Start the Flask Backend (Port 5000)

In your terminal, run:
```bash
python backend/app.py
```
You should see:
```text
==================================================
🚀 SeniorEase AI Backend running on http://127.0.0.1:5000
==================================================
```

### Step 2: Start the Streamlit Frontend (Port 8501)

Open a **second terminal window** and run:
```bash
streamlit run frontend/app.py --server.port 8501
```
Your browser will automatically open at:
`http://localhost:8501`

---

## 📡 API Endpoints

### 1. Health Check
- **URL**: `GET /api/health`
- **Response**:
```json
{
  "service": "SeniorEase AI Backend REST API",
  "status": "healthy",
  "timestamp": "2026-09-19T10:30:00Z"
}
```

### 2. Chat API
- **URL**: `POST /api/chat`
- **Headers**: `Content-Type: application/json`
- **Payload**:
```json
{
  "message": "How do I make text bigger on my phone?",
  "category": "tech"
}
```
- **Response**:
```json
{
  "provider": "mock",
  "response": "📱 **Here are simple steps...**",
  "status": "success",
  "timestamp": "2026-09-19T10:30:00Z"
}
```

---

## 📄 License
MIT License. Free for learning, modification, and community use.
