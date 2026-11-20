
import os
import logging
import threading

import joblib

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =====================================
# LOGGING CONFIGURATION
# =====================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("bolsense")


# =====================================
# MODEL CONFIGURATION
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILENAME = "sentiment_model_v3.joblib"

MODEL_PATH = os.path.join(
    BASE_DIR,
    MODEL_FILENAME
)

MODEL_NAME = "BolSense V3"

MODEL_RUNTIME = "TF-IDF + Logistic Regression"

LABEL_MAPPING = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

inference_lock = threading.Lock()


# =====================================
# FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="BolSense Sentiment API",
    version="3.0.0",
    description=(
        "Lightweight sentiment analysis API "
        "for Hindi, English and Hinglish text."
    )
)


# =====================================
# CORS CONFIGURATION
# =====================================

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "https://bolsense-frontend.onrender.com"
).strip().rstrip("/")

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "https://bolsense-frontend.onrender.com",
]

if FRONTEND_URL and FRONTEND_URL not in ALLOWED_ORIGINS:
    ALLOWED_ORIGINS.append(FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

logger.info("Allowed frontend origins: %s", ALLOWED_ORIGINS)


# =====================================
# REQUEST VALIDATION
# =====================================

class AnalyzeRequest(BaseModel):

    text: str = Field(
        min_length=2,
        max_length=2000
    )


# =====================================
# LOAD TRAINED MODEL
# =====================================

@lru_cache(maxsize=1)
def get_classifier():

    logger.info("Loading BolSense V3 model...")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    logger.info("BolSense V3 loaded successfully.")

    return model


# =====================================
# ROOT ENDPOINT
# =====================================

@app.get("/")
def root():

    return {
        "status": "running",
        "message": "Welcome to BolSense Sentiment API",
        "health": "/api/health",
        "analyze": "/api/analyze"
    }


# =====================================
# HEALTH CHECK
# =====================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "service": "BolSense Sentiment API",
        "model": MODEL_NAME,
        "runtime": MODEL_RUNTIME,
        "model_variant": "V3 - Word and Character TF-IDF"
    }


# =====================================
# SENTIMENT ANALYSIS
# =====================================

@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):

    text = payload.text.strip()

    if len(text) < 2:

        raise HTTPException(
            status_code=400,
            detail="Please enter at least two characters."
        )

    try:

        logger.info("Analyzing text...")

        model = get_classifier()

        with inference_lock:

            prediction = model.predict([text])[0]

            probabilities = model.predict_proba([text])[0]

        predicted_id = int(prediction)

        sentiment = LABEL_MAPPING[predicted_id]

        confidence = float(
            probabilities[predicted_id] * 100
        )

        return {
            "text": text,
            "sentiment": sentiment,
            "raw_label": sentiment,
            "confidence": round(confidence, 2),
            "label_mapping": LABEL_MAPPING,
            "model": MODEL_NAME,
            "disclaimer": (
                "Model predictions are estimates and may not "
                "understand every slang, sarcastic phrase or context."
            )
        }

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception("Analysis failed")

        raise HTTPException(
            status_code=503,
            detail=(
                "The sentiment model could not analyze this text. "
                "Check the backend logs and try again."
            )
        ) from exc
