import os
import sys
import torch
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from models.model_loader import load_model

# Cargar modelo
model, tokenizer = load_model()

def predict_bot(user_data):
    inputs = tokenizer(str(user_data), return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=-1).cpu().item()

def analyze_followers(followers_data):
    df = pd.DataFrame(followers_data)
    df["is_bot"] = df.apply(lambda row: predict_bot(row.to_dict()), axis=1)
    bot_percentage = (df["is_bot"].sum() / len(df)) * 100
    return bot_percentage, df
