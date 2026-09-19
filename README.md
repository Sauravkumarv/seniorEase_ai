# SeniorEase AI 👵👴

**SeniorEase AI** is a simple, accessible, and friendly AI companion designed specifically for senior citizens who may find modern digital tools complex or confusing.

---

## 🔗 Live Shared Link & Deployment

This app is two services: a **Streamlit frontend** and a **Flask backend**. The UI on Streamlit Cloud cannot use `localhost`. Deploy Flask publicly, then point Streamlit at that URL.

GitHub repo: [https://github.com/Sauravkumarv/seniorEase_ai](https://github.com/Sauravkumarv/seniorEase_ai)

### 1. Deploy the Flask backend (Render — free)

1. Push this repo to GitHub (include `Procfile` and `requirements.txt`).
2. Open [https://render.com](https://render.com), sign in with GitHub, and click **New → Web Service**.
3. Select this repository. Use:
   - **Runtime**: Python
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn backend.app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1`
4. Add environment variables (do not put keys in Streamlit):
   ```text
   AI_PROVIDER=gemini
   AI_MODEL=gemini-1.5-flash
   FLASK_DEBUG=False
   GEMINI_API_KEY=your_gemini_key
   AI_API_KEY=your_gemini_key
   ```
5. Deploy. Copy the public URL, for example `https://seniorease-ai-backend.onrender.com`.
6. Open `/api/health` in a browser. You should see `"status": "healthy"`.

Render’s free plan sleeps after idle time. The first request after sleep can take 30–60 seconds.

### 2. Point the existing Streamlit app at the backend

1. Open [https://share.streamlit.io](https://share.streamlit.io) → your app → **⋮ → Settings → Secrets**.
2. Set only the backend URL (Gemini keys stay on Render):
   ```toml
   BACKEND_URL = "https://YOUR-RENDER-SERVICE.onrender.com"
   ```
3. Reboot / redeploy the Streamlit app.

Local development still works with default `http://127.0.0.1:5000` if `BACKEND_URL` is unset.

### ☁️ Streamlit Community Cloud (frontend only)
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
2. Click **"New app"** and select repository: `Sauravkumarv/seniorEase_ai`.
3. Set **Main file path**: `frontend/app.py`.
4. Add the `BACKEND_URL` secret as above, then deploy.

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
