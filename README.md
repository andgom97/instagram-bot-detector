# Instagram Bot Detector

## Descripción del Proyecto
Instagram Bot Detector es una herramienta basada en **DeepSeek LLM** que permite analizar los seguidores de una cuenta de Instagram y determinar cuántos de ellos son bots. Utiliza **Machine Learning** y **scraping con Instaloader** para recopilar datos y clasificarlos con un modelo de IA entrenado.

## Estructura del Proyecto
```
instagram-bot-detector/
│── 📁 src/                    # Código fuente del proyecto
│   ├── 📁 models/             # Modelos y entrenamiento
│   │   ├── deepseek_trainer.py   # Entrenamiento del modelo DeepSeek LLM
│   │   ├── model_loader.py       # Carga del modelo entrenado
│   │   ├── inferencer.py         # Predicciones con el modelo entrenado
│   │   ├── model/                # Carpeta donde se guarda el modelo entrenado
│   │
│   ├── 📁 data/                # Datasets y procesamiento
│   │   ├── dataset_loader.py    # Carga y preprocesamiento del dataset
│   │   ├── scraper.py           # Web scraping de Instagram con Instaloader
│   │
│   ├── 📁 api/                 # Servidor API para consultas
│   │   ├── api_server.py        # FastAPI para consulta de bots
│   │   ├── bot_detector.py      # Lógica de predicción
│   │
│   ├── main.py                 # Script principal
│
│── 📁 notebooks/               # Jupyter notebooks de exploración
│── 📁 tests/                   # Pruebas unitarias
│── 📁 docs/                    # Documentación
│── 📁 logs/                    # Registros de entrenamiento
│── .gitignore                  # Archivos a ignorar en Git
│── README.md                   # Documentación del proyecto
│── requirements.txt             # Librerías necesarias
│── setup.py                     # Script de instalación del paquete
```

## Instalación del Proyecto

Para instalar el proyecto como un paquete de Python, ejecuta:

```bash
pip install -e .
```

Esto instalará el proyecto en modo editable, permitiéndote modificarlo sin necesidad de reinstalarlo.

## Entrenamiento del Modelo
Para entrenar el modelo **DeepSeek LLM**, ejecuta:

```bash
python src/models/deepseek_trainer.py
```

Este script cargará el dataset [InstaFake](https://github.com/fcakyon/instafake-dataset), entrenará el modelo y lo guardará en la carpeta `models/`.

## Uso del `main.py`
El archivo `main.py` permite iniciar la API o analizar un usuario de Instagram desde la línea de comandos.

### Iniciar la API
```bash
python src/main.py --api
```
Esto iniciará el servidor en `http://127.0.0.1:8000`.

### Analizar un usuario de Instagram
```bash
python src/main.py --user usuario_instagram --insta_user tu_usuario --insta_pass tu_contraseña
```
Esto extraerá los seguidores del usuario y calculará el porcentaje de bots.

## Análisis de un Usuario de Instagram con API
Para analizar la cantidad de seguidores bots de un usuario de Instagram mediante la API:

```bash
uvicorn src.api.api_server:app --reload
```

Luego, usa la API desde el navegador o con `curl`:

```bash
curl "http://127.0.0.1:8000/analyze/usuario_instagram"
```

O visita en el navegador:
```
http://127.0.0.1:8000/docs
```
para interactuar con la API usando la interfaz de **FastAPI**.

## 📌 Requisitos Mínimos de Hardware

Para entrenar este modelo de detección de bots en Instagram con **DeepSeek LLM**, se requiere un hardware con suficiente capacidad de cómputo. A continuación, se detallan los requisitos recomendados:

### ✅ Opción 1: Entrenamiento en GPU Local
Si deseas entrenar el modelo en tu propio equipo, asegúrate de contar con:
- **GPU NVIDIA con CUDA** (mínimo **RTX 3090 (24GB VRAM)**, recomendado **A100 (40GB VRAM)**).
- **CPU de alto rendimiento** (mínimo **Intel i7/Ryzen 7**).
- **RAM**: 32GB recomendados (16GB mínimo).
- **Almacenamiento**: 50GB de espacio libre (preferiblemente SSD NVMe).

> **Nota:** Tarjetas gráficas con **menos de 16GB VRAM** (ej. RTX 3060 Laptop) pueden generar errores de memoria (`CUDA out of memory`).

---

### ✅ Opción 2: Entrenamiento en Nube
Si no tienes una GPU potente, puedes usar servicios en la nube con **GPUs de alta VRAM**:

| Plataforma         | GPU Disponible              | Costo Aproximado |
|-------------------|---------------------------|------------------|
| **Google Colab Pro+** | **A100 (40GB VRAM)**      | **$20/mes** |
| **RunPod.io**      | **A100 (40GB), H100 (80GB)** | **$0.50 - $1/h** |
| **Lambda Labs**    | **RTX 4090 (24GB), A100 (80GB)** | **$1.10/h** |
| **AWS EC2**       | **8x A100 (320GB VRAM)**    | **$3 - $5/h** |

> **Recomendación**: Si solo necesitas entrenar el modelo una vez, **Google Colab Pro+** o **RunPod.io** son opciones accesibles y eficientes.

---

### 🔹 Verificación de Hardware
Antes de entrenar, verifica si tu GPU cumple con los requisitos ejecutando:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Contribuciones
Si deseas mejorar el proyecto, siéntete libre de abrir un **Pull Request** o crear un **Issue** en GitHub.

## Licencia
Este proyecto está bajo la licencia **MIT**.

## Autor y Contacto
- **Autor**: [Andrés Gómez Alfonso]
- **Fecha de Creación**: [02, 2025]
- **Email de Contacto**: [andgomalf@gmail.com]

