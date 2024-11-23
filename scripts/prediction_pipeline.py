"""
Pipeline de predicción para nuevos datos.

Este script reutiliza los módulos existentes para:
- Cargar datos.
- Procesar datos (imputación, escalado).
- Cargar el modelo guardado.
- Generar predicciones y probabilidades.
- Guardar los resultados en un archivo CSV.

Dependencias:
    - load_data (load_data.py): Función para cargar datos.
    - preprocess_data (create_features.py): Función para procesar datos.
"""

import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from create_features import preprocess_data  # Reutilización del procesamiento
from load_data import load_data  # Carga de datos desde un CSV

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


if __name__ == "__main__":
    """
    Pipeline principal para realizar predicciones con nuevos datos.
    """
    # Configuración
    INPUT_DATA_PATH = "../data/raw/new_data.csv"  # Ruta del archivo de datos nuevos
    MODEL_PATH = "../models/best_model.pkl"       # Ruta del modelo guardado
    SCALER_PATH = "../artifacts/scaler.pkl"      # Ruta del escalador guardado
    # OUTPUT_PATH = "../data/predictions/predictions.csv"  # Ruta de salida para predicciones

    # Cargar datos
    data = load_data(INPUT_DATA_PATH)

    # Procesar datos reutilizando `preprocess_data` y el escalador existente
    data_processed = preprocess_data(data)
    scaler = joblib.load(SCALER_PATH)
    numeric_cols = data_processed.select_dtypes(include=["number"]).columns
    data_processed[numeric_cols] = scaler.transform(data_processed[numeric_cols])

    # Cargar el modelo
    model = load_model(MODEL_PATH)

    # Realizar predicciones
    predictions, probabilities = predict(model, data_processed)

    # Guardar las predicciones
    # save_predictions(predictions, probabilities, OUTPUT_PATH)

    print("\nPipeline completado exitosamente.")
