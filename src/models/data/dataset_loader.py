import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Obtener la ruta absoluta del directorio raíz del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def load_dataset(fake_data_path=None, real_data_path=None, test_size=0.2, random_state=42):
    """
    Carga y preprocesa el dataset desde archivos JSON, combinando cuentas falsas y reales.

    Args:
        fake_data_path (str): Ruta al dataset de cuentas falsas en JSON.
        real_data_path (str): Ruta al dataset de cuentas genuinas en JSON.
        test_size (float): Porcentaje de datos reservados para pruebas.
        random_state (int): Semilla para la división de datos.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) listas de datos de entrenamiento y prueba.
    """

    # Asegurar que las rutas sean correctas sin doble "src/"
    if fake_data_path is None:
        fake_data_path = os.path.join(BASE_DIR, "models/data/datasets/fake_users.json")
    if real_data_path is None:
        real_data_path = os.path.join(BASE_DIR, "models/data/datasets/genuine_users.json")

    # Normalizar las rutas para evitar errores
    fake_data_path = os.path.normpath(fake_data_path)
    real_data_path = os.path.normpath(real_data_path)

    # Verificar que los archivos existen
    if not os.path.exists(fake_data_path):
        raise FileNotFoundError(f"❌ Dataset de usuarios falsos no encontrado en {fake_data_path}")
    if not os.path.exists(real_data_path):
        raise FileNotFoundError(f"❌ Dataset de usuarios reales no encontrado en {real_data_path}")

    # Cargar JSON en DataFrames
    with open(fake_data_path, "r", encoding="utf-8") as f:
        fake_users = json.load(f)

    with open(real_data_path, "r", encoding="utf-8") as f:
        real_users = json.load(f)

    df_fake = pd.DataFrame(fake_users)
    df_real = pd.DataFrame(real_users)

    # Agregar etiquetas para clasificación
    df_fake["isFake"] = 1  # 1 = Bot
    df_real["isFake"] = 0  # 0 = Real

    # Combinar datasets
    df = pd.concat([df_fake, df_real], ignore_index=True)

    # Seleccionar características
    feature_columns = [
        "userFollowerCount", "userFollowingCount", "userBiographyLength", "userMediaCount",
        "userHasProfilPic", "userIsPrivate", "usernameDigitCount", "usernameLength"
    ]

    # **🔹 Convertir columnas a float para evitar el FutureWarning**
    df[feature_columns] = df[feature_columns].astype(float)

    # **🔹 Asegurar que X sea una copia para evitar `SettingWithCopyWarning`**
    X = df[feature_columns].copy()  # 🔹 Se hace una copia explícita
    y = df["isFake"].copy()  # 🔹 También se copia `y` por seguridad

    # Normalización de datos numéricos usando `.loc[]` para evitar el warning
    scaler = MinMaxScaler()
    X.loc[:, feature_columns] = scaler.fit_transform(X[feature_columns])

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Prueba para verificar si la ruta del dataset es correcta
    print("Ruta esperada para fake_users.json:", os.path.normpath(os.path.join(BASE_DIR, "models/data/datasets/fake_users.json")))
    print("Ruta esperada para genuine_users.json:", os.path.normpath(os.path.join(BASE_DIR, "models/data/datasets/genuine_users.json")))

    try:
        X_train, X_test, y_train, y_test = load_dataset()
        print(f"✅ Datos de entrenamiento: {X_train.shape[0]} muestras")
        print(f"✅ Datos de prueba: {X_test.shape[0]} muestras")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
