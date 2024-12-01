"""
Pipeline de predicción para nuevos datos.

Este script reutiliza los módulos existentes para:
- Cargar y balancear datos.
- Procesar datos (imputación, escalado).
- Cargar el modelo guardado.
- Generar predicciones y probabilidades.
- Guardar los resultados en un archivo CSV.

Dependencias:
    - load_and_balance_data (load_data.py): Función para cargar y balancear datos.
    - preprocess_data (create_features.py): Función para procesar datos.
    - scale_and_save_scaler (create_features.py): Para escalar datos.
"""

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from load_data import load_and_balance_data  # Carga y balanceo de datos
from create_features import preprocess_data  # Procesamiento de datos
from create_models import load_data  # Para cargar datasets procesados

def load_model(model_path: str):
    """
    Carga el modelo entrenado desde un archivo.

    Args:
        model_path (str): Ruta del archivo del modelo.

    Returns:
        object: Modelo cargado.
    """
    model = joblib.load(model_path)
    print(f"Modelo cargado desde: {model_path}")
    return model


def predict(model, data: pd.DataFrame) -> tuple:
    """
    Genera predicciones utilizando el modelo cargado.

    Args:
        model (object): Modelo entrenado.
        data (pd.DataFrame): Datos procesados para predicción.

    Returns:
        tuple: Predicciones y probabilidades.
    """
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1] if hasattr(model, "predict_proba") else None
    return predictions, probabilities


def save_predictions(predictions, probabilities, output_path: str):
    """
    Guarda las predicciones y probabilidades en un archivo CSV.

    Args:
        predictions (np.ndarray): Predicciones generadas por el modelo.
        probabilities (np.ndarray): Probabilidades generadas por el modelo.
        output_path (str): Ruta donde se guardarán los resultados.

    Returns:
        None
    """
    results = pd.DataFrame({
        "Predicción": predictions,
        "Probabilidad": probabilities if probabilities is not None else None
    })
    results.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en: {output_path}")
