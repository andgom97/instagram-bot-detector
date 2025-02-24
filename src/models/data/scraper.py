import instaloader
import re

def get_instagram_data(username):
    """
    Extracts Instagram profile data using Instaloader.

    Args:
        username (str): Instagram username.

    Returns:
        dict: A dictionary containing extracted numerical features.
    """
    loader = instaloader.Instaloader()

    try:
        print(f"🔍 Fetching data for {username}...")
        profile = instaloader.Profile.from_username(loader.context, username)

        # Extract profile details
        followers = profile.followers
        following = profile.followees
        bio_length = len(profile.biography) if profile.biography else 0
        posts = profile.mediacount
        has_profile_pic = int(profile.has_profile_pic)  # Convert to 0/1
        is_private = int(profile.is_private)  # Convert to 0/1
        username_digit_count = len(re.findall(r"\d", username))  # Count digits in username
        username_length = len(username)

        # Return extracted features
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
        print(f"❌ Error: Profile '{username}' does not exist.")
        return None
    except instaloader.exceptions.ConnectionException as e:
        print(f"❌ Connection error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def extract_user_data(username):
    """
    Scrapes Instagram to extract numerical + text-based user features.

    Args:
    username (str): Instagram username.

    Returns:
    tuple: A tuple containing extracted text and numerical features.

    """
    data = get_instagram_data(username)

    if data:
        print("\n✅ Extracted Instagram Data:")
        for key, value in data.items():
            print(f"{key}: {value}")

        text_features = f"User has {data['followers']} followers, follows {data['following']} accounts, " \
                        f"has a biography of {data['bio_length']} characters, posted {data['posts']} media items, " \
                        f"{'has' if data['has_profile_pic'] else 'does not have'} a profile picture, " \
                        f"{'has' if data['is_private'] else 'does not have'} a private account, " \
                        f"username contains {data['digit_count']} digits and has {data['username_length']} characters."

        numerical_features = [
            data["followers"] / (data["following"] + 1),  # Follower-to-Following Ratio
            1 if data["digit_count"] > 0 else 0,  # Has numbers in username
            (data["posts"] + 1) / (data["followers"] + 1)  # Engagement Score
        ]

    else:
        print("❌ Failed to fetch data.")
        
    return text_features, numerical_features
