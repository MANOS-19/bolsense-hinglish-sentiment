
import os
import logging
import threading
import hashlib
import shutil
import tempfile
import urllib.request

from functools import lru_cache

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =====================================
# LOGGING CONFIGURATION
# =====================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hinglish-sentiment")


# =====================================
# MODEL CONFIGURATION
# =====================================

MODEL_NAME = "airzipm/sentiment-analysis-muril-v2"

GITHUB_RELEASE_URL = (
    "https://github.com/MANOS-19/"
    "bolsense-hinglish-sentiment/releases/download/v1.0.0"
)

MODEL_FILENAME = "model_last4_int8.onnx"
TOKENIZER_FILENAME = "tokenizer.json"

MODEL_URL = f"{GITHUB_RELEASE_URL}/{MODEL_FILENAME}"
TOKENIZER_URL = f"{GITHUB_RELEASE_URL}/{TOKENIZER_FILENAME}"

MODEL_SHA256 = "24a7b6980f783604697e0d01a44b4b3d03636c67930d8444edec9fca7a63995"

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
    title="Hinglish Sentiment API",
    version="1.0.0",
    description=(
        "Multilingual sentiment analysis API "
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
# DOWNLOAD HELPER
# =====================================

def download_file(url, destination, expected_sha256=None):

    temporary_path = destination + ".part"

    try:

        logger.info("Downloading: %s", os.path.basename(destination))

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "BolSense-Sentiment-API/1.0"
            }
        )

        sha256 = hashlib.sha256()
        downloaded_bytes = 0

        with urllib.request.urlopen(request, timeout=60) as response:

            with open(temporary_path, "wb") as output:

                while True:

                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    output.write(chunk)
                    sha256.update(chunk)
                    downloaded_bytes += len(chunk)

        if downloaded_bytes == 0:
            raise RuntimeError("Downloaded file is empty.")

        if expected_sha256:

            actual_sha256 = sha256.hexdigest()

            if actual_sha256.lower() != expected_sha256.lower():

                raise RuntimeError(
                    "SHA-256 checksum verification failed."
                )

        os.replace(temporary_path, destination)

        logger.info(
            "Download completed: %s (%.2f MB)",
            os.path.basename(destination),
            downloaded_bytes / (1024 * 1024)
        )

    except Exception:

        if os.path.exists(temporary_path):
            os.remove(temporary_path)

        raise


# =====================================
# DOWNLOAD MODEL FILES FROM GITHUB
# =====================================

def download_model_files():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # First check for local model files.
    local_model_path = os.path.join(
        base_dir,
        "onnx_model",
        MODEL_FILENAME
    )

    local_tokenizer_path = os.path.join(
        base_dir,
        "onnx_model",
        TOKENIZER_FILENAME
    )

    if (
        os.path.exists(local_model_path)
        and os.path.exists(local_tokenizer_path)
    ):

        logger.info("Using locally stored MuRIL ONNX model.")

        return local_model_path, local_tokenizer_path

    # Render deployment: download files from GitHub Releases.
    cache_dir = os.path.join(
        tempfile.gettempdir(),
        "bolsense_model_cache"
    )

    os.makedirs(
        cache_dir,
        exist_ok=True
    )

    model_path = os.path.join(
        cache_dir,
        MODEL_FILENAME
    )

    tokenizer_path = os.path.join(
        cache_dir,
        TOKENIZER_FILENAME
    )

    # Download model if missing or checksum is incorrect.
    model_valid = False

    if os.path.exists(model_path):

        logger.info("Checking cached ONNX model...")

        sha256 = hashlib.sha256()

        with open(model_path, "rb") as model_file:

            for chunk in iter(
                lambda: model_file.read(1024 * 1024),
                b""
            ):
                sha256.update(chunk)

        model_valid = (
            sha256.hexdigest().lower()
            == MODEL_SHA256.lower()
        )

    if not model_valid:

        logger.info("Downloading MuRIL ONNX model from GitHub...")

        download_file(
            MODEL_URL,
            model_path,
            expected_sha256=MODEL_SHA256
        )

    else:

        logger.info("Valid cached ONNX model found.")

    # Download tokenizer if missing or empty.
    if (
        not os.path.exists(tokenizer_path)
        or os.path.getsize(tokenizer_path) == 0
    ):

        logger.info("Downloading tokenizer from GitHub...")

        download_file(
            TOKENIZER_URL,
            tokenizer_path
        )

    else:

        logger.info("Cached tokenizer found.")

    logger.info("Model files are ready.")

    return model_path, tokenizer_path


# =====================================
# LOAD ONNX MODEL AND TOKENIZER
# =====================================

@lru_cache(maxsize=1)
def get_classifier():

    logger.info("Loading optimized MuRIL ONNX model...")

    model_path, tokenizer_path = download_model_files()

    tokenizer = Tokenizer.from_file(
        tokenizer_path
    )

    tokenizer.enable_truncation(
        max_length=128
    )

    tokenizer.enable_padding(
        length=128,
        pad_id=0,
        pad_token="[PAD]"
    )

    options = ort.SessionOptions()

    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    logger.info("Optimized MuRIL model loaded successfully.")

    return tokenizer, session


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
        "model": MODEL_NAME,
        "runtime": "ONNX Runtime",
        "model_variant": "Last 6 Layers INT8"
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

        print(
            "ANALYZE ENDPOINT REACHED",
            flush=True
        )

        tokenizer, session = get_classifier()

        encoded = tokenizer.encode(
            text[:2000]
        )

        inputs = {
            "input_ids": np.array(
                [encoded.ids],
                dtype=np.int64
            ),
            "attention_mask": np.array(
                [encoded.attention_mask],
                dtype=np.int64
            ),
            "token_type_ids": np.array(
                [encoded.type_ids],
                dtype=np.int64
            )
        }

        with inference_lock:

            outputs = session.run(
                None,
                inputs
            )

        logits = outputs[0][0]

        exp_logits = np.exp(
            logits - np.max(logits)
        )

        probabilities = (
            exp_logits / np.sum(exp_logits)
        )

        predicted_id = int(
            np.argmax(probabilities)
        )

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

        print(
            "ACTUAL ERROR:",
            repr(exc),
            flush=True
        )

        logger.exception("Analysis failed")

        raise HTTPException(
            status_code=503,
            detail=(
                "The sentiment model could not analyze this text. "
                "Check the backend logs and try again."
            )
        ) from exc
