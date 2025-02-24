import os
import sys
import joblib
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Ensure dataset_loader.py can be imported correctly
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(BASE_DIR, "src"))

from data.dataset_loader import load_dataset

# 🚀 Load dataset (Both text & numerical features)
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = load_dataset()

# 🚀 Debug: Check dataset structure
print("\n🔍 Dataset Type Check:")
print(f"   - X_train_text type: {type(X_train_text)}, shape: {X_train_text.shape}")
print(f"   - X_train_num type: {type(X_train_num)}, shape: {X_train_num.shape}")
print(f"   - y_train type: {type(y_train)}, shape: {y_train.shape}")

# 🚀 Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)  # 🔹 Increase TF-IDF features for better accuracy
X_train_text_tfidf = vectorizer.fit_transform(X_train_text)
X_test_text_tfidf = vectorizer.transform(X_test_text)

# 🚀 Combine TF-IDF features with numerical features
X_train_combined = hstack([X_train_text_tfidf, X_train_num])  # Combine sparse TF-IDF and dense numerical features
X_test_combined = hstack([X_test_text_tfidf, X_test_num])

# 🚀 Debug: Print dataset shape before applying SMOTE
print("\n✅ Dataset before SMOTE balancing:")
print(f"   - X_train_combined shape: {X_train_combined.shape}")
print(f"   - y_train shape: {y_train.shape}")

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)

# 🚀 Debug: Print dataset shape after SMOTE
print("\n✅ Dataset balanced using SMOTE.")
print(f"   - Original dataset: {X_train_combined.shape[0]} samples")
print(f"   - Resampled dataset: {X_train_resampled.shape[0]} samples")

# 🚀 Define Hyperparameter Grid for GridSearch
param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [3, 6, 9],  
    'learning_rate': [0.01, 0.1, 0.2],  
    'subsample': [0.8, 1.0],  
    'colsample_bytree': [0.8, 1.0]  
}

# 🚀 Perform Grid Search to find the best hyperparameters
print("\n🔍 Running Grid Search for best hyperparameters...")
grid_search = GridSearchCV(
    XGBClassifier(objective="binary:logistic"),
    param_grid,
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1
)
grid_search.fit(X_train_resampled, y_train_resampled)

# 🚀 Use best model from Grid Search
model = grid_search.best_estimator_
print(f"\n✅ Best Hyperparameters: {grid_search.best_params_}")

# 🚀 Train XGBoost Classifier with Best Hyperparameters
print("\n🚀 Training XGBoost Classifier...")
model.fit(X_train_resampled, y_train_resampled)

# 🚀 Evaluate Model
y_pred = model.predict(X_test_combined)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Training Completed. Accuracy: {accuracy * 100:.2f}%")
print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))

# 🚀 Ensure the model directory exists before saving
model_dir = os.path.join(BASE_DIR, "src/models/model")
os.makedirs(model_dir, exist_ok=True)

# 🚀 Save the Model and Vectorizer
model_save_path = os.path.join(model_dir, "xgboost_model.pkl")
vectorizer_save_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

joblib.dump(model, model_save_path)
joblib.dump(vectorizer, vectorizer_save_path)

print(f"\n✅ Model saved to `{model_save_path}`")
print(f"✅ Vectorizer saved to `{vectorizer_save_path}`")
