import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Asegurar que se puede importar dataset_loader.py correctamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data.dataset_loader import load_dataset

# Cargar los datos usando la nueva función corregida
X_train, X_test, y_train, y_test = load_dataset()

# Definir el nombre del modelo
model_name = "deepseek-ai/deepseek-llm-7b-base"

# Configurar carpetas dentro de `src/models/`
offload_path = os.path.join(os.path.dirname(__file__), "offload_weights")
os.makedirs(offload_path, exist_ok=True)

save_path = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(save_path, exist_ok=True)

# Verificar si hay GPU disponible
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32  # FP16 en GPU, FP32 en CPU

# **🔹 Cargar el modelo correctamente sin moverlo manualmente**
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="sequential",  # 🔹 Distribuye el modelo sin usar toda la memoria de la GPU
    offload_folder=offload_path,  # 🔹 Guarda pesos offloaded en disco si es necesario
    torch_dtype=torch_dtype,  # 🔹 FP16 en GPU para reducir memoria
)

print("✅ Modelo cargado correctamente con `device_map=sequential`.")

# **🔹 Configurar entrenamiento SIN intentar mover el modelo**
training_args = TrainingArguments(
    output_dir=save_path,  # Guarda los modelos en `src/models/model/`
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=1,  # 🔹 USAR BATCH SIZE PEQUEÑO PARA EVITAR OOM
    per_device_eval_batch_size=1,
    num_train_epochs=2,  # 🔹 MENOS ÉPOCAS PARA REDUCIR MEMORIA
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True if torch.cuda.is_available() else False,  # 🔹 Mantener FP16 en GPU para reducir memoria
    save_total_limit=1,  # 🔹 Mantener solo 1 checkpoint para reducir uso de disco
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,  # 🔹 Reduce consumo de memoria con optimización
    optim="adamw_torch",  # 🔹 Usa AdamW optimizado para memoria en Torch
    push_to_hub=False,  # No subir a Hugging Face Hub
)

# **🔹 Crear el entrenador SIN intentar mover el modelo**
trainer = Trainer(
    model=model,  # **No mover manualmente, `device_map="sequential"` lo maneja**
    args=training_args,
    train_dataset=X_train,
    eval_dataset=X_test,
)

# **🔹 Iniciar el entrenamiento**
trainer.train()

# **🔹 Guardar el modelo entrenado en `src/models/model/`**
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ Entrenamiento completado y modelo guardado en {save_path}")
