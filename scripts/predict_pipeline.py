"""
Script para cargar un pipeline ya entrenado, realizar predicciones con datos de prueba
y registrar los resultados en MLflow.
"""

import os
import pickle
import json
from datetime import datetime
import pandas as pd
import mlflow

# Configuración inicial
PROJECT_PATH = os.path.abspath(os.path.join(os.getcwd()))
DATA_PROCESSED_PATH = os.path.join(PROJECT_PATH, "data", "processed")
PREDICTIONS_PATH = os.path.join(PROJECT_PATH, "data", "predictions")
ARTIFACTS_PATH = os.path.join(PROJECT_PATH, "artifacts")
PIPELINE_PATH = os.path.join(ARTIFACTS_PATH, "best_model_pipeline.pkl")
MODEL_RESULTS_PATH = os.path.join(ARTIFACTS_PATH, "model_results.json")

# Crear carpeta de predicciones si no existe
os.makedirs(PREDICTIONS_PATH, exist_ok=True)


# Funciones Utilitarias
def cargar_pipeline(pipeline_path):
    """
    Carga el pipeline entrenado desde un archivo .pkl.

    Args:
        pipeline_path (str): Ruta del archivo del pipeline entrenado.

    Returns:
        Pipeline: Objeto pipeline cargado.
    """
    try:
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)
        print("Pipeline cargado correctamente.")
        return pipeline
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"El archivo del pipeline no existe en: {pipeline_path}") from exc


def cargar_datos_prueba(data_path):
    """
    Carga los datos de prueba desde la ruta especificada.

    Args:
        data_path (str): Ruta de los datos procesados.

    Returns:
        tuple: DataFrames de X_test e y_test.
    """
    try:
        x_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"))
        print(f"Datos de prueba cargados: X_test {x_test.shape}, y_test {y_test.shape}")
        return x_test, y_test
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Error al cargar los datos de prueba: {exc}") from exc


def guardar_predicciones(predictions, true_labels, output_path):
    """
    Guarda las predicciones en un archivo CSV.

    Args:
        predictions (array): Predicciones del modelo.
        true_labels (array): Etiquetas verdaderas.
        output_path (str): Ruta donde guardar el archivo CSV.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = f"predictions_{timestamp}.csv"
    output_file = os.path.join(output_path, file_name)

    predictions_df = pd.DataFrame({
        "True_Label": true_labels,
        "Predicted_Label": predictions
    })

    predictions_df.to_csv(output_file, index=False)
    print(f"Predicciones guardadas en: {output_file}")
    return output_file


def registrar_predicciones_mlflow(file_path):
    """
    Registra las predicciones en MLflow.

    Args:
        file_path (str): Ruta del archivo de predicciones.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Predicciones_Modelo")

    with mlflow.start_run():
        mlflow.log_artifact(file_path, artifact_path="predictions")
        print("Predicciones registradas en MLflow.")


def registrar_modelos_mlflow(results_path):
    """
    Registra los resultados de los modelos entrenados en MLflow.

    Args:
        results_path (str): Ruta del archivo JSON con los resultados de los modelos.
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"El archivo de resultados no existe en: {results_path}")

    with open(results_path, "r", encoding="utf-8") as file:
        results = json.load(file)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Resultados_Modelos")

    for model_name, model_data in results.items():
        with mlflow.start_run(run_name=model_name):
            # Log de métricas
            mlflow.log_metric("f1_score", model_data["f1_score"])

            # Log de hiperparámetros
            if model_data["hyperparameters"]:
                for param, value in model_data["hyperparameters"].items():
                    mlflow.log_param(param, value)

            # Log del modelo
            model_path = model_data["model_path"]
            mlflow.log_artifact(model_path, artifact_path="modelos")
            print(f"Modelo {model_name} registrado en MLflow.")


# Función Principal
def main():
    """
    Función principal del script.
    """
    # Registrar modelos en MLflow
    print("\nRegistrando modelos en MLflow...")
    registrar_modelos_mlflow(MODEL_RESULTS_PATH)

    # Cargar pipeline entrenado
    print("\nCargando el pipeline entrenado...")
    pipeline = cargar_pipeline(PIPELINE_PATH)

    # Cargar datos de prueba
    print("\nCargando datos de prueba...")
    x_test, y_test = cargar_datos_prueba(DATA_PROCESSED_PATH)

    # Realizar predicciones
    print("\nRealizando predicciones...")
    y_pred = pipeline.predict(x_test)

    # Guardar predicciones
    print("\nGuardando predicciones en archivo...")
    output_file = guardar_predicciones(y_pred, y_test.values.ravel(), PREDICTIONS_PATH)

    # Registrar predicciones en MLflow
    print("\nRegistrando predicciones en MLflow...")
    registrar_predicciones_mlflow(output_file)

    print("\nProceso completado con éxito.")


if __name__ == "__main__":
    main()
