from setuptools import setup, find_packages

setup(
    name="instagram-bot-detector",
    version="1.0.0",
    author="Andres Gomez Alfonso",
    author_email="andgomalf@gmail.com",
    description="Herramienta para detectar bots en Instagram usando XGBoost y otras técnicas de Machine Learning",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/andgom97/instagram-bot-detector",  # Reemplázalo con tu URL de GitHub si es público
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "Flask==3.1.0",
        "ImageHash==4.3.2",
        "imbalanced_learn==0.13.0",
        "imblearn==0.0",
        "instaloader==4.14.1",
        "joblib==1.4.2",
        "numpy==2.2.3",
        "pandas==2.2.3",
        "Pillow==11.1.0",
        "Requests==2.32.3",
        "scikit_learn==1.6.1",
        "scipy==1.15.2",
        "setuptools==65.5.0",
        "torch==2.5.1+cu121",
        "tqdm==4.67.1",
        "xgboost==2.1.4"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "instagram-bot-detector=src.main:main",  # CLI command
        ],
    },
)
