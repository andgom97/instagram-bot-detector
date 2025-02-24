from model_loader import load_model

# Cargar modelo y vectorizador
model, vectorizer = load_model()

def predict_user(text):
    """
    Predict whether an Instagram user is a bot or real based on text input.
    """
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return "Bot Detected" if prediction == 1 else "Real User"
