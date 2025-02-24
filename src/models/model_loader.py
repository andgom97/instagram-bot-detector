import os
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Rutas de los archivos del modelo
model_path = os.path.join(BASE_DIR, "src/models/model/xgboost_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "src/models/model/tfidf_vectorizer.pkl")

# Cargar modelo y vectorizador
def load_model():
    print("🔄 Loading trained XGBoost model...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("✅ Model and vectorizer loaded successfully!")
    return model, vectorizer
