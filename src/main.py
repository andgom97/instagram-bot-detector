import argparse
import logging
from api.api_server import app
from data.scraper import get_followers_data
from api.bot_detector import analyze_followers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_api():
    """Boots up the API server."""
    logger.info("🚀 Starting API server at http://0.0.0.0:8000")
    app.run(debug=True, host="0.0.0.0", port=8000)

def analyze_user(username, user, password):
    """Fetches the followers of an Instagram user and analyzes how many are bots."""
    print(f"Analyzing followers of {username}...")
    followers = get_followers_data(username, user, password)
    bot_percentage, df = analyze_followers(followers)
    print(f"Percentage of instagram bots in {username} followers: {bot_percentage:.2f}%")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instagram Bot Detector")
    parser.add_argument("--api", action="store_true", help="Boots up the API server")
    parser.add_argument("--user", type=str, help="Instagram User to analyze")
    parser.add_argument("--insta_user", type=str, help="Your username for Instagram (for login)")
    parser.add_argument("--insta_pass", type=str, help="Your password for Instagram (for login)")
    
    args = parser.parse_args()

    if args.api:
        run_api()
    elif args.user and args.insta_user and args.insta_pass:
        analyze_user(args.user, args.insta_user, args.insta_pass)
    else:
        print("Use --api to start the server or --user to analyze a user.")
