from setuptools import setup, find_packages

setup(
    name="instagram-bot-detector",
    version="1.0.0",
    author="Andres Gomez Alfonso",
    author_email="andgomalf@gmail.com",
    description="Herramienta para detectar bots en Instagram usando DeepSeek LLM",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/andgom97/instagram-bot-detector",  # Reemplázalo con tu URL de GitHub si es público
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "scikit-learn",
        "pandas",
        "instaloader",
        "fastapi",
        "uvicorn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "bot-detector-api=api.api_server:app",  # Para ejecutar la API como un comando
        ],
    },
)
