import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from data.dataset_loader import preprocess_single_sample

# Configuración del dispositivo (GPU si está disponible, de lo contrario, CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar modelo y tokenizador entrenados
model_path = "./models/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

def predict_bot(user_data):
    """
    Realiza una inferencia para determinar si una cuenta es bot o real.

    Args:
        user_data (dict): Datos del usuario con características como número de seguidores, seguidos, publicaciones, etc.

    Returns:
        dict: Resultado con la predicción y la probabilidad de que sea un bot.
    """

    # Preprocesar los datos del usuario
    processed_data = preprocess_single_sample(user_data)

    # Tokenizar entrada
    inputs = tokenizer(
        str(processed_data), truncation=True, padding="max_length", max_length=128, return_tensors="pt"
    ).to(device)

    # Realizar la predicción
    with torch.no_grad():
        outputs = model(**inputs)

    # Extraer la predicción (0 = real, 1 = bot)
    prediction = torch.argmax(outputs.logits, dim=-1).cpu().item()

    return {
        "is_bot": bool(prediction),
        "prediction_label": "Bot" if prediction == 1 else "Real",
        "confidence": torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy().tolist()
    }

if __name__ == "__main__":
    sample_user = {
        "num_followers": 150,
        "num_following": 200,
        "num_posts": 5,
        "has_profile_pic": 1,
        "has_biography": 0
    }

    result = predict_bot(sample_user)
    print(f"Predicción: {result}")
