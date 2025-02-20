import torch
print(torch.cuda.get_device_name(0))  # Nombre de tu GPU

# Verificar si la GPU está siendo usada
if torch.cuda.is_available():
    print(f"✅ PyTorch está usando la GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria de la GPU usada: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
else:
    print("⚠️ PyTorch está usando la CPU. Verifica si CUDA está instalado correctamente.")
