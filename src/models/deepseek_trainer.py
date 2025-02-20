from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from data.dataset_loader import load_dataset
import torch

# Configurar el dispositivo para el modelo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar y preprocesar el dataset
X_train, X_test, y_train, y_test = load_dataset()

# Tokenizador y modelo DeepSeek
model_name = "deepseek-ai/deepseek-llm-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Tokenizar los datos
def tokenize_function(texts):
    return tokenizer(texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

train_encodings = tokenize_function(X_train.to_json())
test_encodings = tokenize_function(X_test.to_json())

# Crear dataset en formato compatible con Hugging Face Trainer
import torch
from torch.utils.data import Dataset

class InstagramBotDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = InstagramBotDataset(train_encodings, y_train.tolist())
test_dataset = InstagramBotDataset(test_encodings, y_test.tolist())

# Definir entrenamiento
training_args = TrainingArguments(
    output_dir="./models/model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Entrenar
trainer.train()
model.save_pretrained("./models/model")
tokenizer.save_pretrained("./models/model")

print("Entrenamiento completado y modelo guardado en ./models/model")
