import numpy as np
from scipy.sparse import hstack
from model_loader import load_model

# Load the trained model and vectorizer
model, vectorizer = load_model()

def predict_user(text, numerical_features):
    """
    Predict whether an Instagram user is a bot or a real user.

    Args:
        text (str): The structured text description of the user's Instagram activity.
        numerical_features (list): A list of numerical features extracted from the user's profile.

    Returns:
        str: "Bot Detected" if the prediction is 1, otherwise "Real User".
    """

    # Convert text to TF-IDF vector
    text_vectorized = vectorizer.transform([text])

    # Convert numerical features to a NumPy array
    num_features = np.array(numerical_features).reshape(1, -1)

    # Combine TF-IDF with numerical features
    input_data = hstack([text_vectorized, num_features])

    # Make a prediction
    prediction = model.predict(input_data)[0]

    return "Bot Detected" if prediction == 1 else "Real User"

