import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def convert_to_text(row):
    """Convert numerical features into a structured text format for TF-IDF transformation."""
    return f"User has {row['userFollowerCount']} followers, follows {row['userFollowingCount']} accounts, " \
           f"has a biography of {row['userBiographyLength']} characters, posted {row['userMediaCount']} media items, " \
           f"{'has' if row['userHasProfilPic'] else 'does not have'} a profile picture, " \
           f"{'has' if row['userIsPrivate'] else 'does not have'} a private account, " \
           f"username contains {row['usernameDigitCount']} digits and has {row['usernameLength']} characters."

def load_dataset():
    """Load and preprocess the dataset with both numerical and text-based features."""

    fake_data_path = os.path.join(BASE_DIR, "models/data/datasets/fake_users.json")
    real_data_path = os.path.join(BASE_DIR, "models/data/datasets/genuine_users.json")

    # Ensure the dataset files exist
    if not os.path.exists(fake_data_path):
        raise FileNotFoundError(f"❌ Dataset not found: {fake_data_path}")
    if not os.path.exists(real_data_path):
        raise FileNotFoundError(f"❌ Dataset not found: {real_data_path}")

    # Load JSON into DataFrames
    df_fake = pd.read_json(fake_data_path)
    df_real = pd.read_json(real_data_path)

    # Assign labels (1 = Bot, 0 = Real User)
    df_fake["isFake"] = 1
    df_real["isFake"] = 0

    # Combine datasets
    df = pd.concat([df_fake, df_real], ignore_index=True)

    # Define numerical feature columns
    feature_columns = [
        "userFollowerCount", "userFollowingCount", "userBiographyLength", "userMediaCount",
        "userHasProfilPic", "userIsPrivate", "usernameDigitCount", "usernameLength"
    ]

    # Ensure no missing values before processing
    df = df.dropna(subset=feature_columns + ["isFake"])
    
    # Convert labels to integers
    df["isFake"] = df["isFake"].astype(int)

    ### 🚀 Add New Numerical Features 🚀 ###
    
    # 1️⃣ **Follower-to-Following Ratio**
    df["follower_following_ratio"] = df["userFollowerCount"] / (df["userFollowingCount"] + 1)  # Avoid division by zero

    # 2️⃣ **Binary Feature: Does the username contain numbers?**
    df["has_numbers_in_username"] = df["usernameDigitCount"].apply(lambda x: 1 if x > 0 else 0)

    # 3️⃣ **Engagement Score (Posts per Follower)**
    df["engagement_score"] = (df["userMediaCount"] + 1) / (df["userFollowerCount"] + 1)  # Avoid division by zero

    # Update feature columns to include new ones
    feature_columns += ["follower_following_ratio", "has_numbers_in_username", "engagement_score"]

    # Normalize numerical features (Scale between 0 and 1)
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    # Convert dataset into structured text format (for TF-IDF transformation)
    df["text"] = df.apply(convert_to_text, axis=1)

    # Split dataset into training and testing sets
    X_train_text, X_test_text, y_train, y_test = train_test_split(df["text"], df["isFake"], test_size=0.2, random_state=42)
    X_train_num, X_test_num, _, _ = train_test_split(df[feature_columns], df["isFake"], test_size=0.2, random_state=42)

    # Drop any remaining NaN values (just in case)
    X_train_text.dropna(inplace=True)
    X_test_text.dropna(inplace=True)
    y_train = y_train.dropna()
    y_test = y_test.dropna()

    # Ensure labels are 1D arrays for XGBoost
    y_train = np.array(y_train).reshape(-1)
    y_test = np.array(y_test).reshape(-1)

    # 🚀 Debugging: Print dataset shapes before returning
    print(f"\n✅ Final dataset shapes:")
    print(f"   - X_train_text: {X_train_text.shape}")
    print(f"   - X_train_num: {X_train_num.shape}")
    print(f"   - y_train: {y_train.shape}")
    print(f"   - X_test_text: {X_test_text.shape}")
    print(f"   - X_test_num: {X_test_num.shape}")
    print(f"   - y_test: {y_test.shape}")

    return X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test
