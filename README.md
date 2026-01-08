# Instagram Bot Detector

## Project Description

Instagram Bot Detector is a tool based on **XGBoost** that analyzes an Instagram account’s followers and determines how many of them are bots. It uses **Machine Learning** and **scraping with Instaloader** to collect data and classify them using a trained AI model.

## Project Structure
```
instagram-bot-detector/
│── 📁 src/                    # Project source code
│   ├── 📁 models/             # Models and training
│   │   ├── xgboost_trainer.py    # XGBoost model training
│   │   ├── model_loader.py       # Loads the trained model
│   │   ├── inferencer.py         # Predictions using the trained model
│   │   ├── model/                # Folder where the trained model is stored
│   │
│   ├── 📁 data/                 # Datasets and processing
│   │   ├── dataset_loader.py    # Loads and preprocesses the dataset
│   │   ├── scraper.py           # Instagram web scraping with Instaloader
│   │   ├── 📁 datasets/         # Datasets in JSON format for model training
│   │
│   ├── 📁 api/                 # API server for queries
│   │   ├── api_server.py        # Flask API for bot detection
│   │   ├── bot_detector.py      # Prediction logic
│   │
│   ├── main.py                 # Main script
│
│── 📁 notebooks/               # Jupyter notebooks for exploration
│── 📁 tests/                   # Unit tests
│── 📁 docs/                    # Documentation
│── 📁 logs/                    # Training logs
│── .gitignore                  # Files to ignore in Git
│── README.md                   # Project documentation
│── requirements.txt             # Required libraries
│── setup.py                     # Package installation script
```

## Project Installation

To install the project as a Python package, run:

```bash
pip install -e .
```

This installs the project in editable mode, allowing modifications without reinstallation.

## Model Training
To train the **XGBoost** model first you will need to place your datasets (you can use the [InstaFake](https://github.com/fcakyon/instafake-dataset)) into the `src/data/datasets/` path in `.json` format, and then execute:

```bash
python src/models/xgboost_trainer.py
```

This script loads the datasets defined in `src/data/datasets/`, trains the model, and saves it in the `models/` folder.

## Using `main.py`

The `main.py` script allows you to either start the API or analyze an Instagram user from the command line.

### Start the API

```bash
python src/main.py --api
```

This starts the server at `http://127.0.0.1:8000`.

### Analyze an Instagram User

```bash
python src/main.py --user usuario_instagram --insta_user tu_usuario --insta_pass tu_contraseña
```
This extracts the user's followers and calculates the percentage of bots.

## Analyze an Instagram User via API

To analyze the number of bot followers of an Instagram user through the API:

```bash
python src/api/api_server.py

```

Then, use the API from your browser or via `curl`:

```bash
curl "http://127.0.0.1:8000/analyze/test_user?insta_user=your_instagram_username&insta_pass=your_instagram_password"

```

Or visit in your browser:
```
http://127.0.0.1:8000/docs
```
to interact with the API using the **FlaskAPI** interface.

## Contributions
If you would like to improve the project, feel free to open a **Pull Request** or create an **Issue** on GitHub.

## Author & Contact
- **Author**: [Andrés Gómez Alfonso]
- **Creation Date**: [02, 2025]
- **Contact Email**: [andgomalf@gmail.com]

