# Mano's Full-Stack Projects

This bundle contains two independent applications:

1. **WeatherNow** — React + Vite frontend, Express backend, Open-Meteo API.
2. **Hinglish Sentiment Analyzer** — React + Vite frontend, FastAPI backend, Hugging Face multilingual sentiment model.

Both projects are complete starter applications with frontend, backend, setup instructions, error handling, and responsive styling.

## Requirements
- Node.js 20+ and npm
- Python 3.10+
- Internet connection for weather data and first-time model download

## 1. WeatherNow

Open two terminals.

### Backend
```powershell
cd weather-now\backend
npm install
npm run dev
```
Backend: http://localhost:5000

### Frontend
```powershell
cd weather-now\frontend
npm install
npm run dev
```
Frontend: http://localhost:5173

No API key is required. The backend uses Open-Meteo geocoding and forecast services.

## 2. Hinglish Sentiment Analyzer

Open two terminals.

### Backend
```powershell
cd hinglish-sentiment\backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Backend: http://localhost:8000

The first prediction may take longer because Hugging Face downloads the multilingual model. The selected model is `lxyuan/distilbert-base-multilingual-cased-sentiments-student`.

### Frontend
```powershell
cd hinglish-sentiment\frontend
npm install
npm run dev
```
Frontend: http://localhost:5174

## Notes
- The sentiment model is multilingual and can handle Hindi and English text, including many Hinglish inputs. Like all models, it can misclassify slang, sarcasm, code-switching, and context-dependent phrases.
- Do not present the confidence score as a guarantee of correctness.
- These apps are independent; each has its own frontend and backend.
