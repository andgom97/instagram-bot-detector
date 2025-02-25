import datetime
import math
import os
import random
import time
from typing import List
from flask import json
import instaloader
import re
import requests
from PIL import Image
import imagehash
from io import BytesIO
from tqdm import tqdm
import yaml

# Load configuration from config.yml
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config/config.yml"))
with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)

INSTA_USER = config["insta_user"]
INSTA_PASS = config["insta_pass"]

# Default Instagram profile picture URLs (Common placeholders)
DEFAULT_PROFILE_PIC_HASHES = {
    "instagram_light": "ffffc3c3e7e78181",  # Example hash for Instagram's default picture (light mode)
    "instagram_dark": "c381183c3c18bdff"   # Example hash for dark mode
}

def get_image_hash(image_url):
    """Downloads an image and returns its hash for comparison."""
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("L")  # Convert to grayscale
        return str(imagehash.average_hash(img))  # Generate a hash of the image
    except Exception as e:
        print(f"❌ Error fetching image: {e}")
        return None

def has_custom_profile_pic(profile_pic_url):
    """Checks if a user has a custom profile picture by comparing against Instagram's default."""
    image_hash = get_image_hash(profile_pic_url)

    if image_hash is None:
        return False  # Assume no custom picture if the image couldn't be retrieved
    print(f"image hash: {image_hash}")
    return image_hash not in DEFAULT_PROFILE_PIC_HASHES.values()  # Check if it's different from default

def get_instagram_data(user: instaloader.Profile):
    """
    Extracts Instagram profile data using Instaloader, including checking if the profile picture is custom.
    """
    loader = instaloader.Instaloader()

    try:
        print(f"🔍 Fetching data for {user.username}...")
        profile = instaloader.Profile.from_username(loader.context, user.username)

        # Extract profile details
        followers = profile.followers
        following = profile.followees
        bio_length = len(profile.biography) if profile.biography else 0
        posts = profile.mediacount
        profile_pic_url = profile.profile_pic_url  # Profile picture URL
        is_private = int(profile.is_private)  # Convert to 0/1
        username_digit_count = len(re.findall(r"\d", user.username))  # Count digits in username
        username_length = len(user.username)

        # Check if the profile has a custom picture
        has_profile_pic = int(has_custom_profile_pic(profile_pic_url))

        return {
            "followers": followers,
            "following": following,
            "bio_length": bio_length,
            "posts": posts,
            "has_profile_pic": has_profile_pic,
            "is_private": is_private,
            "digit_count": username_digit_count,
            "username_length": username_length
        }

    except instaloader.exceptions.ProfileNotExistsException:
        print(f"❌ Error: Profile '{user.username}' does not exist.")
        return None
    except instaloader.exceptions.ConnectionException as e:
        print(f"❌ Connection error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def get_followers_data(username: str, attempt: int = 1, retrieved_followers: list = None):
    """
    Fetches the list of followers for a given Instagram username while handling rate limits.
    
    - Resumes fetching if interrupted.
    - Uses exponential backoff for delays.

    Args:
        username (str): The target Instagram username.
        attempt (int): Retry attempt count.
        retrieved_followers (list): Followers collected so far.

    Returns:
        list(str): A list of follower usernames.
    """
    loader = instaloader.Instaloader()

    try:
        # ✅ Login to Instagram
        loader.login(INSTA_USER, INSTA_PASS)

        print(f"🔍 Fetching followers of {username} (Attempt {attempt})...")
        profile = instaloader.Profile.from_username(loader.context, username)

        # ✅ Initialize list if first call
        if retrieved_followers is None:
            retrieved_followers = []

         # ✅ Get total number of followers
        total_followers = profile.followers
        print(total_followers)

        # ✅ Initialize tqdm progress bar
        with tqdm(total=total_followers, desc="Fetching followers", unit="follower") as pbar:
            # ✅ Get followers and resume from last retrieved follower
            for follower in profile.get_followers():
                if follower.username in retrieved_followers:
                    continue  # ✅ Skip already retrieved followers
                
                retrieved_followers.append(follower.username)

                # ✅ Get follower data and save to JSON
                follower_data = get_instagram_data(follower)
                if follower_data:
                    save_follower_data(username, follower_data)

                # ✅ Update progress bar
                pbar.update(1)

            # ✅ Add random delay to prevent detection
            time.sleep(random.uniform(3, 10))  

        print(f"✅ Retrieved {len(retrieved_followers)} followers.")
        return retrieved_followers  # ✅ Return the full list

    except instaloader.exceptions.ConnectionException:
        wait_time = min(300, math.pow(2, attempt) * 60)  # ✅ Exponential backoff (max 5 min)
        print(f"❌ Rate limit reached. Waiting {wait_time / 60:.1f} minutes before retrying...")
        time.sleep(wait_time)

        # ✅ Retry while keeping already retrieved followers
        return get_followers_data(username, attempt + 1, retrieved_followers)

def save_follower_data(username: str, follower_data: dict):
    """
    Saves follower data to a JSON file in the src/data/datasets directory.
    
    Args:
        username (str): The target Instagram username.
        follower_data (dict): Data of the follower to save.
    """
    timestamp = datetime.now().strftime("%d-%m-%YT%H-%M-%S")
    filename = f"{username}-{timestamp}.json"
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", filename))

    # Load existing data if the file exists
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            data = json.load(file)
    else:
        data = []

    # Append new follower data
    data.append(follower_data)

    # Save updated data to the file
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)

    print(f"✅ Saved data for follower {follower_data['username']} to {filename}")