"""
Script principal para ejecutar las diferentes fases del pipeline de datos.

Este script permite realizar las siguientes tareas:
- Cargar datos (`load_data`).
- Procesar datos (`create_features`).
- Entrenar y evaluar modelos (`create_models`).
- Realizar predicciones (`prediction_pipeline`).

Dependencias:
    - load_data (scripts/load_data.py)
    - create_features (scripts/create_features.py)
    - create_models (scripts/create_models.py)
    - prediction_pipeline (scripts/prediction_pipeline.py)
"""

import argparse
import sys
from scripts.load_data import load_data
from scripts.create_features import preprocess_data, scale_and_save_scaler
from scripts.create_models import train_and_evaluate_models, grid_search_best_model, save_best_model
from scripts.prediction_pipeline import load_model, predict, save_predictions
import pandas as pd


def main():
    """
    Punto de entrada principal para ejecutar las diferentes fases del pipeline.
    """
    parser = argparse.ArgumentParser(description="Pipeline de procesamiento, modelado y predicción.")
    parser.add_argument(
        "task",
        choices=["load_data", "create_features", "create_models", "prediction_pipeline"],
        help="Tarea a ejecutar: load_data, create_features, create_models, prediction_pipeline",
    )
    args = parser.parse_args()

    # Configuración de rutas comunes
    INPUT_DATA_PATH = "../data/raw/creditcard.csv"
    PROCESSED_DATA_PATH = "../data/processed/"
    MODEL_PATH = "../models/best_model.pkl"
    SCALER_PATH = "../artifacts/scaler.pkl"
    PREDICTION_INPUT_PATH = "../data/raw/new_data.csv"
    PREDICTION_OUTPUT_PATH = "../data/predictions/predictions.csv"

    if args.task == "load_data":
        # Cargar datos
        print("\n=== Ejecutando: Cargar Datos ===")
        data = load_data(INPUT_DATA_PATH)
        print(f"Datos cargados con éxito. Dimensiones: {data.shape}")

    elif args.task == "create_features":
        # Procesar datos
        print("\n=== Ejecutando: Procesar Datos ===")
        data = load_data(INPUT_DATA_PATH)
        data_processed = preprocess_data(data)
        scale_and_save_scaler(data_processed, PROCESSED_DATA_PATH, SCALER_PATH)
        print(f"Datos procesados guardados en: {PROCESSED_DATA_PATH}")

    elif args.task == "create_models":
        # Entrenar y evaluar modelos
        print("\n=== Ejecutando: Entrenamiento de Modelos ===")
        from sklearn.model_selection import train_test_split

        # Cargar datos procesados
        X_train = pd.read_csv(PROCESSED_DATA_PATH + "X_train.csv")
        y_train = pd.read_csv(PROCESSED_DATA_PATH + "y_train.csv")
        X_test = pd.read_csv(PROCESSED_DATA_PATH + "X_test.csv")
        y_test = pd.read_csv(PROCESSED_DATA_PATH + "y_test.csv")

        best_model, best_model_name, models_with_params = train_and_evaluate_models(X_train, y_train, X_test, y_test)
        best_model = grid_search_best_model(X_train, y_train, best_model_name, models_with_params)
        save_best_model(best_model, MODEL_PATH)
        print(f"Modelo guardado en: {MODEL_PATH}")

    elif args.task == "prediction_pipeline":
        # Realizar predicciones
        print("\n=== Ejecutando: Pipeline de Predicción ===")
        from scripts.prediction_pipeline import preprocess_input_data

        model = load_model(MODEL_PATH)
        data = preprocess_input_data(PREDICTION_INPUT_PATH, SCALER_PATH)
        predictions, probabilities = predict(model, data)
        save_predictions(predictions, probabilities, PREDICTION_OUTPUT_PATH)
        print(f"Predicciones guardadas en: {PREDICTION_OUTPUT_PATH}")

    else:
        print("Tarea no reconocida. Use --help para más información.")
        sys.exit(1)


if __name__ == "__main__":
    main()
