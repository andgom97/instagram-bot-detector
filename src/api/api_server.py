import os
import sys
import logging
from flask import Flask, jsonify, request

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
    - insta_user (str): Instagram login username (provided in request).
    - insta_pass (str): Instagram login password (provided in request).

    Returns:
    - dict: Analysis results including bot percentage and follower predictions.
    """
    try:
        logger.info(f"🔍 Fetching followers of {username}...")

        # ✅ Get Instagram credentials from request
        insta_user = request.args.get("insta_user")
        insta_pass = request.args.get("insta_pass")

        # ✅ Validate credentials
        if not insta_user or not insta_pass:
            logger.error("❌ Missing Instagram credentials in request.")
            return jsonify({"error": "Instagram username and password are required."}), 400

        # ✅ Fetch followers using login credentials
        followers = get_followers_data(username, insta_user, insta_pass)
        if not followers:
            logger.error(f"❌ Error fetching followers for {username}.")
            return jsonify({"error": "User not found or private."}), 404

        logger.info(f"✅ Retrieved {len(followers)} followers for {username}. Analyzing bot percentage...")

        # ✅ Analyze followers to detect bots
        bot_percentage, df = analyze_followers(followers)

        return jsonify({
            "username": username,
            "bot_percentage": f"{bot_percentage:.2f}%",
            "followers_analyzed": len(df),
            "follower_predictions": df.to_dict(orient="records")
        })

    except Exception as e:
        logger.error(f"❌ Error analyzing {username}: {str(e)}")
        return jsonify({"error": "Internal server error."}), 500

def start_api():
    """Function to start FastAPI server with Uvicorn."""
    logger.info("🚀 Starting API server at http://0.0.0.0:8000")
    app.run(debug=True, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_api()