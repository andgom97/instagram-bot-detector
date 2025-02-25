import pandas as pd
from models.model_loader import load_model
from data.scraper import get_instagram_data
from models.inferencer import predict_user

# Load the trained XGBoost model and vectorizer
model, vectorizer = load_model()

def analyze_followers(followers):
    """
    Analyzes a list of Instagram followers and calculates the percentage of bots.

    Args:
        followers (list): List of Instagram usernames.

    Returns:
        tuple: (bot_percentage, DataFrame with follower predictions)
    """
    bot_count = 0
    follower_details = []

    for follower in followers:
        follower_data = get_instagram_data(follower)  # Extract profile info

        if not follower_data:
            continue  # Skip if data retrieval fails

        # Generate text features for TF-IDF
        text_features = f"User has {follower_data['followers']} followers, follows {follower_data['following']} accounts, " \
                        f"has a biography of {follower_data['bio_length']} characters, posted {follower_data['posts']} media items, " \
                        f"{'has' if follower_data['has_profile_pic'] else 'does not have'} a profile picture, " \
                        f"{'has' if follower_data['is_private'] else 'does not have'} a private account, " \
                        f"username contains {follower_data['digit_count']} digits and has {follower_data['username_length']} characters."

        # Extract numerical features
        numerical_features = [
            follower_data["followers"] / (follower_data["following"] + 1),  # Follower-to-Following Ratio
            1 if follower_data["digit_count"] > 0 else 0,  # Has numbers in username
            (follower_data["posts"] + 1) / (follower_data["followers"] + 1)  # Engagement Score
        ]

        # Make a prediction using the model
        prediction = predict_user(text_features, numerical_features)

        # Track bot count
        if prediction == "Bot Detected":
            bot_count += 1

        # Store follower details
        follower_details.append({
            "username": follower,
            "prediction": prediction,
            "profile_data": follower_data
        })

    # Convert follower details into a DataFrame
    df = pd.DataFrame(follower_details)

    # Calculate bot percentage
    bot_percentage = (bot_count / len(followers)) * 100 if followers else 0

    return bot_percentage, df
