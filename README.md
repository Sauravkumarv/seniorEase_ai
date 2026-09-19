# SeniorEase AI 👵👴

**SeniorEase AI** is a simple, accessible, and friendly AI companion designed specifically for senior citizens who may find modern digital tools complex or confusing.

---

## 🔗 Live Shared Link & Deployment

### 🌐 Instant Shareable Public URL (Live Now)
- **Public Shared URL**: [https://twenty-memes-glow.loca.lt](https://twenty-memes-glow.loca.lt)
- **Tunnel Password (if prompted by loca.lt)**: `104.134.37.8`

### ☁️ Permanent Free Hosting (Streamlit Community Cloud)
Your codebase is ready on GitHub: [https://github.com/Sauravkumarv/seniorEase_ai](https://github.com/Sauravkumarv/seniorEase_ai)

To deploy a permanent 24/7 link:
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
2. Click **"New app"** and select repository: `Sauravkumarv/seniorEase_ai`.
3. Set **Main file path**: `frontend/app.py`.
4. Click **Advanced Settings** -> **Secrets** and add:
   ```toml
   AI_PROVIDER = "gemini"
   GEMINI_API_KEY = "your_key_here"
   ```
5. Click **Deploy!**

---

## 🌟 Key Features

- **Senior-Friendly Frontend**: Large readable fonts (22px+), high-contrast colors, touch-friendly buttons, and calm professional styling.
- **RESTful Flask Backend**: Runs on port `5000` with modular routing, input validation, and clear JSON API responses.
- **Patient & Clear AI Assistant**: System prompts and responses tailored to deliver empathetic, step-by-step guidance.
- **Proactive Assistance**: Includes a `"You may also need"` section for required items (IDs, dates, payment methods).
- **"Explain Something Simply"**: Paste complex notices, messages, or legal text for simplified explanations.
- **Privacy & Safety Filter**: Automatically detects and intercepts sensitive data (OTP, PIN, passwords).
- **Quick Help Topics**: One-click quick questions for WhatsApp, Banking, Train Booking, Email, Shopping, and Government Services.

---

## 🛠 Tech Stack

- **Python**: Core programming language
- **Flask**: Backend REST API (`port 5000`)
- **Streamlit**: Frontend User Interface (`port 8501`)
- **Google Gemini API**: `gemini-1.5-flash` model integration
- **requests**: Frontend-to-Backend HTTP communication
- **python-dotenv**: Environment variable management

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
│   │   └── chat.py           # Chat, explain, and health API routes
│   └── services/
│       └── ai_service.py     # AI service layer with Gemini/OpenAI & safety filter
│
├── .env                      # Environment configuration file
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 📄 License
MIT License. Free for learning, modification, and community use.
