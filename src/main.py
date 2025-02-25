import argparse
import logging
from api.api_server import app
from data.scraper import get_followers_data
from api.bot_detector import analyze_followers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_api():
    """Inicia el servidor API usando FastAPI y Uvicorn."""
    logger.info("🚀 Starting API server at http://0.0.0.0:8000")
    app.run(debug=True, host="0.0.0.0", port=8000)

def analyze_user(username, user, password):
    """Obtiene los seguidores de un usuario y analiza cuántos son bots."""
    print(f"Analizando seguidores de {username}...")
    followers = get_followers_data(username, user, password)
    bot_percentage, df = analyze_followers(followers)
    print(f"Porcentaje de seguidores bots: {bot_percentage:.2f}%")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instagram Bot Detector")
    parser.add_argument("--api", action="store_true", help="Inicia la API de Flask")
    parser.add_argument("--user", type=str, help="Usuario de Instagram a analizar")
    parser.add_argument("--insta_user", type=str, help="Tu usuario de Instagram (para login)")
    parser.add_argument("--insta_pass", type=str, help="Tu contraseña de Instagram (para login)")
    
    args = parser.parse_args()

    if args.api:
        run_api()
    elif args.user and args.insta_user and args.insta_pass:
        analyze_user(args.user, args.insta_user, args.insta_pass)
    else:
        print("Usa --api para iniciar el servidor o --user para analizar un usuario.")
