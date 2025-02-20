import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(fake_data_path="data/fake_users.csv", real_data_path="data/genuine_users.csv", test_size=0.2, random_state=42):
    """
    Carga y preprocesa el dataset InstaFake, combinando cuentas falsas y reales.

    Args:
        fake_data_path (str): Ruta al dataset de cuentas falsas.
        real_data_path (str): Ruta al dataset de cuentas genuinas.
        test_size (float): Porcentaje de datos reservados para pruebas.
        random_state (int): Semilla para la división de datos.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) listas de datos de entrenamiento y prueba.
    """
    
    # Cargar datasets
    df_fake = pd.read_csv(fake_data_path)
    df_real = pd.read_csv(real_data_path)

    # Agregar etiquetas (1 = bot, 0 = real)
    df_fake["label"] = 1
    df_real["label"] = 0

    # Unir los datasets
    df = pd.concat([df_fake, df_real], ignore_index=True)

    # Seleccionar características relevantes
    features = ["num_followers", "num_following", "num_posts"]
    X = df[features]
    y = df["label"]

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Prueba de carga del dataset
    X_train, X_test, y_train, y_test = load_dataset()
    print(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Datos de prueba: {X_test.shape[0]} muestras")
