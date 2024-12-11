"""
Script para realizar ingeniería de características.

Este módulo contiene funciones para:
- Manejo de valores faltantes.
- Codificación de variables categóricas.
- Selección de características.
- División en conjuntos de entrenamiento y prueba.
- Escalado de características numéricas.

Funciones:
    - preprocess_data: Realiza el preprocesamiento inicial de los datos.
    - select_features: Selecciona las características relevantes.
    - scale_and_save_scaler: Escala los datos y guarda el escalador como un artefacto.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza el preprocesamiento de datos:
    - Manejo de valores faltantes.
    - Codificación de variables categóricas.

    Args:
        data (pd.DataFrame): Dataset original.

    Returns:
        pd.DataFrame: Dataset preprocesado.
    """
    # Manejo de valores faltantes
    if data.isnull().sum().sum() > 0:
        data.fillna(data.mean(), inplace=True)
        print("Valores faltantes imputados con la media.")

    # Codificación de variables categóricas
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        print(f"Se aplicó One-Hot Encoding a: {list(categorical_cols)}")

    return data


def select_features(data: pd.DataFrame, target_col: str, corr_threshold: float = 0.01) -> pd.DataFrame:
    """
    Selecciona características relevantes basándose en correlación:
    - Elimina columnas con baja correlación con la variable objetivo.
    - Elimina características redundantes altamente correlacionadas entre sí.

    Args:
        data (pd.DataFrame): Dataset completo (incluyendo la variable objetivo).
        target_col (str): Nombre de la columna objetivo.
        corr_threshold (float): Umbral de correlación mínima con la variable objetivo.

    Returns:
        pd.DataFrame: Dataset con las características seleccionadas.
    """
    correlation_matrix = data.corr()

    # Eliminar características con baja correlación con la variable objetivo
    low_corr_features = correlation_matrix[target_col][correlation_matrix[target_col] < corr_threshold].index
    print(f"Características eliminadas por baja correlación: {list(low_corr_features)}")
    data = data.drop(columns=low_corr_features, errors="ignore")

    # Eliminar características redundantes altamente correlacionadas
    redundant_features = set()
    for col in correlation_matrix.columns:
        if col == target_col:
            continue
        high_corr = correlation_matrix[col][correlation_matrix[col] > 0.8].index.drop(col)
        redundant_features.update(high_corr)

    print(f"Características eliminadas por alta redundancia: {list(redundant_features)}")
    data = data.drop(columns=redundant_features, errors="ignore")

    return data


def scale_and_save_scaler(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_path: str) -> tuple:
    """
    Escala los datos y guarda el escalador como un artefacto para uso futuro.

    Args:
        X_train (pd.DataFrame): Conjunto de características de entrenamiento.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        scaler_path (str): Ruta donde se guardará el escalador.

    Returns:
        tuple: X_train_scaled, X_test_scaled
    """
    numeric_cols = X_train.select_dtypes(include=["number"]).columns

    # Instanciar StandardScaler
    scaler = StandardScaler()

    # Escalar datos
    X_train_scaled = X_train.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Guardar el escalador como artefacto
    joblib.dump(scaler, scaler_path)
    print(f"Escalador guardado en {scaler_path}")

    return X_train_scaled, X_test_scaled