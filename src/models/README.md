# Model Training & Inference Architecture

```mermaid
graph TD;
    
    %% Data Preparation
    A[Dataset Loader] -->|Load JSON Data| B[Text Features]
    A -->|Load JSON Data| C[Numerical Features]
    
    %% Text Processing
    B -->|Apply TF-IDF Vectorization| D[Text Vectorized Features]
    
    %% Numerical Processing
    C -->|Keep as Numeric Values| E[Numerical Features Processed]
    
    %% Combine Features
    D -->|Merge| F[Combined Features]
    E -->|Merge| F

    %% Data Balancing
    F -->|Apply SMOTE| G[Balanced Data]
    
    %% Hyperparameter Optimization
    G -->|Run Grid Search| H[Optimal XGBoost Parameters]

    %% Model Training
    H -->|Train XGBoost Model| I[Trained Model]
    
    %% Save Model & Vectorizer
    I -->|Save XGBoost Model| J[model_loader.py]
    D -->|Save TF-IDF Vectorizer| J

    %% Model Inference
    J -->|Load Model & Vectorizer| K[Inference Engine]
    K -->|Predict User Profile| L[Bot or Real User]

---

## **Explanation of Workflow**
1. **Data Loading (`dataset_loader.py`)**  
   - Loads **text-based** and **numerical** features from JSON.
   - Text data is processed via **TF-IDF vectorization**.
   - Numerical features are **kept as is** for direct processing.

2. **Feature Engineering**
   - **TF-IDF transformation** is applied to text.
   - **Numerical data is processed** separately.
   - Both are **merged into a single dataset**.

3. **Data Balancing with SMOTE**
   - Since real and bot accounts may be imbalanced, **SMOTE** (Synthetic Minority Over-sampling Technique) is applied.

4. **Hyperparameter Optimization**
   - `GridSearchCV` is used to find the best hyperparameters for **XGBoost**.

5. **Model Training (`xgboost_trainer.py`)**
   - The **optimized XGBoost model** is trained.
   - The **trained model** and **TF-IDF vectorizer** are saved to disk.

6. **Model Loading (`model_loader.py`)**
   - Loads the saved **XGBoost model** and **TF-IDF vectorizer**.

7. **Inference (`inferencer.py`)**
   - Converts **new user input (text & numerical features)** into the required format.
   - Applies **TF-IDF to text** and **merges numerical features**.
   - Runs a **prediction using the trained model**.
   - Returns `"Bot Detected"` or `"Real User"`.

---
