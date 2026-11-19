import os
import logging
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline


# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hinglish-sentiment")


# Model configuration
MODEL_NAME = "airzipm/sentiment-analysis-muril-v2"


# FastAPI application
app = FastAPI(
    title="Hinglish Sentiment API",
    version="1.0.0",
    description="Multilingual sentiment analysis API for Hindi, English and Hinglish text."
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

if FRONTEND_URL not in ALLOWED_ORIGINS:
    ALLOWED_ORIGINS.append(FRONTEND_URL)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================
# REQUEST VALIDATION
# =====================================

class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=2, max_length=2000)


# =====================================
# LOAD SENTIMENT MODEL
# =====================================

@lru_cache(maxsize=1)
def get_classifier():

    logger.info("Loading sentiment model: %s", MODEL_NAME)

    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME
    )


# =====================================
# ROOT ENDPOINT
# =====================================

@app.get("/")
def root():

    return {
        "status": "running",
        "message": "Welcome to Hinglish Sentiment API",
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
        "service": "Hinglish Sentiment API",
        "model": MODEL_NAME
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

        classifier = get_classifier()

        result = classifier(
            text[:2000],
            truncation=True
        )[0]

        raw = str(result["label"]).lower()

        # Sentiment classification
        if "positive" in raw or raw in ("label_2", "2"):

            sentiment: Literal[
                "Positive", "Negative", "Neutral"
            ] = "Positive"

        elif "negative" in raw or raw in ("label_0", "0"):

            sentiment = "Negative"

        else:

            sentiment = "Neutral"

        return {
            "text": text,
            "sentiment": sentiment,
            "raw_label": result["label"],
            "confidence": round(
                float(result["score"]) * 100, 2
            ),
            "label_mapping": classifier.model.config.id2label,
            "model": MODEL_NAME,
            "disclaimer": (
                "Model predictions are estimates and may not "
                "understand every slang, sarcastic phrase or context."
            )
        }

    except Exception as exc:

        logger.exception("Analysis failed")

        raise HTTPException(
            status_code=503,
            detail=(
                "The sentiment model could not analyze this text. "
                "Check the backend logs and try again."
            )
        ) from exc