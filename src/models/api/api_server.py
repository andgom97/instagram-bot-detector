import os
import sys
from fastapi import FastAPI, HTTPException
from .bot_detector import analyze_followers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from data.scraper import get_followers

import uvicorn
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Instagram Bot Detector API",
    description="API para detectar bots en los seguidores de una cuenta de Instagram.",
    version="1.0.0"
)

@app.get("/")
def root():
    """Ruta raíz para verificar que la API está activa."""
    return {"message": "Instagram Bot Detector API está corriendo."}

@app.get("/analyze/{username}")
def analyze_user(username: str, insta_user: str, insta_pass: str):
    """
    Obtiene los seguidores de un usuario de Instagram y analiza cuántos son bots.

    - `username`: Usuario de Instagram a analizar.
    - `insta_user`: Usuario de Instagram para autenticación.
    - `insta_pass`: Contraseña del usuario de Instagram para autenticación.
    """
    try:
        logger.info(f"Analizando seguidores de {username}...")
        followers = get_followers(username, insta_user, insta_pass)
        
        if not followers:
            raise HTTPException(status_code=404, detail="No se encontraron seguidores para este usuario.")

        bot_percentage, df = analyze_followers(followers)

        return {
            "username": username,
            "bot_percentage": f"{bot_percentage:.2f}%",
            "followers_analyzed": len(df),
            "followers_details": df.to_dict(orient="records")
        }

    except Exception as e:
        logger.error(f"Error analizando {username}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")

def start_api():
    """Función para iniciar el servidor FastAPI con Uvicorn."""
    logger.info("Iniciando servidor API en http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    start_api()

