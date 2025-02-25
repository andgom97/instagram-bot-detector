from http.client import HTTPException
import os
import sys
import logging
from flask import Flask, jsonify

# ✅ Explicitly ensure `src/` is in Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))

# Import load_model function from model_loader.py
from api.bot_detector import analyze_followers
from data.scraper import get_followers_data
from models.model_loader import load_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load XGBoost model and TF-IDF vectorizer at startup
logger.info("🔄 Loading XGBoost model and TF-IDF vectorizer...")
model, vectorizer = load_model()
logger.info("✅ Model loaded successfully!")

@app.route("/")
def root():
    """Root endpoint to verify API status."""
    return jsonify({"message": "Instagram Bot Detector API is running."})

@app.route("/analyze/<username>", methods=["GET"])
def analyze_user(username: str):
    """
    Fetches the followers of an Instagram user and analyzes how many are bots.
    
    Args:
    - username (str): Instagram username to analyze.
    - insta_user (str): Your Instagram username.
    - insta_pass (str): Your Instagram password.

    Returns:
    - dict: Analysis results including bot percentage and follower
    predictions.
    """
    try:
        logger.info(f"🔍 Fetching followers of {username}...")

        followers = get_followers_data(username)
        if not followers:
            logger.error(f"❌ Error fetching followers for {username}.")
            raise HTTPException(status_code=404, detail="User not found or private.")

        logger.info(f"✅ Retrieved {username} followers. Analyzing bot percentage...") 
        bot_percentage, df = analyze_followers(followers)

        return jsonify({
            "username": username,
            "bot_percentage": f"{bot_percentage:.2f}%",
            "followers_analyzed": len(df),
            "follower_predictions": df.to_dict(orient="records")
        })

    except Exception as e:
        logger.error(f"❌ Error analyzing {username}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")

def start_api():
    """Function to start FastAPI server with Uvicorn."""
    logger.info("🚀 Starting API server at http://0.0.0.0:8000")
    app.run(debug=True, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    analyze_user("pergo.dj")
    # start_api()