import os
import sys
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from .bot_detector import analyze_followers

# Ensure we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data.scraper import get_followers
from models.model_loader import load_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Instagram Bot Detector API",
    description="API for detecting bots among Instagram followers using XGBoost.",
    version="2.0.0"
)

# Load XGBoost model and TF-IDF vectorizer at startup
logger.info("🔄 Loading XGBoost model and TF-IDF vectorizer...")
model, vectorizer = load_model()
logger.info("✅ Model loaded successfully!")

@app.get("/")
def root():
    """Root endpoint to verify API status."""
    return {"message": "Instagram Bot Detector API is running."}

@app.get("/analyze/{username}")
def analyze_user(username: str, insta_user: str, insta_pass: str):
    """
    Fetches the followers of an Instagram user and analyzes how many are bots.

    - `username`: Instagram username to analyze.
    - `insta_user`: Instagram username for authentication.
    - `insta_pass`: Instagram password for authentication.
    """
    try:
        logger.info(f"🔍 Fetching followers of {username}...")
        followers = get_followers(username, insta_user, insta_pass)

        if not followers:
            raise HTTPException(status_code=404, detail="No followers found for this user.")

        logger.info(f"✅ Retrieved {len(followers)} followers. Analyzing with XGBoost...")

        bot_percentage, df = analyze_followers(followers, model, vectorizer)

        return {
            "username": username,
            "bot_percentage": f"{bot_percentage:.2f}%",
            "followers_analyzed": len(df),
            "followers_details": df.to_dict(orient="records")
        }

    except Exception as e:
        logger.error(f"❌ Error analyzing {username}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")

def start_api():
    """Function to start FastAPI server with Uvicorn."""
    logger.info("🚀 Starting API server at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    start_api()