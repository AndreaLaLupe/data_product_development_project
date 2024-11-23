"""
Script para realizar ingeniería de características.

Este módulo contiene funciones para:
- Manejo de valores faltantes.
- Codificación de variables categóricas.
- Selección de características.
- División en conjuntos de entrenamiento y prueba.
- Escalado de características numéricas.
- Manejo del desbalanceo con SMOTE.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
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
        if col in redundant_features or col == target_col:
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


def handle_imbalance(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Aplica SMOTE para manejar el desbalance de clases en los datos de entrenamiento.

    Args:
        X_train (pd.DataFrame): Conjunto de características de entrenamiento.
        y_train (pd.Series): Etiquetas de entrenamiento.

    Returns:
        tuple: X_train_resampled, y_train_resampled
    """
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("SMOTE aplicado. Distribución de clases después del balanceo:")
    print(Counter(y_train_resampled))

    return X_train_resampled, y_train_resampled
