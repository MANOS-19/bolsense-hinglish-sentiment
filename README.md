# BolSense – Hinglish Sentiment Analyzer

An AI-powered web application that analyzes the sentiment of Hindi-English mixed (Hinglish) text using a multilingual NLP model. It classifies user input into Positive, Negative, or Neutral sentiment and provides a confidence score for each prediction.

## Overview

BolSense helps understand the emotions and opinions expressed through everyday Hinglish communication. Users can enter Hindi, English, or mixed-language sentences and receive instant sentiment predictions through an interactive web interface.

## Features

- **Multilingual Sentiment Analysis:** Supports Hindi, English, and Hinglish text.
- **Three Sentiment Categories:** Classifies text as Positive, Negative, or Neutral.
- **Confidence Score:** Displays the model's prediction confidence.
- **Interactive Interface:** Clean and responsive React-based design.
- **Example Inputs:** Provides sample sentences for quick testing.
- **Copy Results:** Allows users to copy their analysis results.
- **REST API Integration:** Connects the React frontend with a FastAPI backend.

## Tech Stack

| Category | Technologies |
|---|---|
| Frontend | React.js, JavaScript, CSS3 |
| Build Tool | Vite |
| Backend | Python, FastAPI |
| Machine Learning | Hugging Face Transformers |
| NLP Model | DistilBERT Multilingual |
| API Testing | Swagger UI |
| Version Control | Git, GitHub |

## AI Model

BolSense uses the pretrained multilingual sentiment classification model:

`lxyuan/distilbert-base-multilingual-cased-sentiments-student`

The model predicts one of three sentiment categories:

- Positive
- Neutral
- Negative

The application also displays the prediction confidence returned by the model.

## Project Structure

```text
bolsense-hinglish-sentiment/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── .gitignore
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── style.css
│   │
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── .gitignore
└── README.md
```

## Installation and Setup

### Prerequisites

- Python 3.10+
- Node.js and npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/MANOS-19/bolsense-hinglish-sentiment.git

cd bolsense-hinglish-sentiment
```

### 2. Backend Setup

Open a terminal:

```powershell
cd backend

python -m venv venv

.\venv\Scripts\activate

pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn main:app --reload --port 8000
```

Backend URL:

`http://localhost:8000`

### 3. Frontend Setup

Open another terminal:

```bash
cd frontend

npm install

npm run dev
```

Frontend URL:

`http://localhost:5174`

### 4. API Documentation

FastAPI provides interactive API documentation through Swagger UI.

`http://localhost:8000/docs`

Use the `/api/analyze` POST endpoint to submit text for sentiment analysis.

## Example

**Input:**

```text
Yeh movie bahut achi hai, I really loved it!
```

**Output:**

```json
{
  "sentiment": "Positive",
  "confidence": 99.21
}
```

*Confidence values are illustrative and may vary between predictions.*

## Important Note

The multilingual model provides estimated sentiment predictions. It may not correctly interpret every slang expression, sarcastic sentence, or context-dependent Hinglish phrase. Confidence scores should not be treated as guarantees of prediction accuracy.

## Developer

**Mano S**

B.Tech – Artificial Intelligence and Machine Learning

GitHub: https://github.com/MANOS-19

---

**BolSense – Understanding the feeling behind the words.**