"""
Script principal para ejecutar las diferentes fases del pipeline de datos.

Este script permite realizar las siguientes tareas:
- Cargar y balancear datos (`load_data`).
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
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from scripts.load_data import load_and_balance_data
from scripts.create_features import preprocess_data, scale_and_save_scaler, select_features
from scripts.create_models import train_and_evaluate_models, grid_search_best_model, save_best_model
from scripts.prediction_pipeline import load_model, predict, save_predictions


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
    RAW_DATA_PATH = "../data/raw/creditcard.csv"
    BALANCED_DATA_PATH = "../data/interim/creditcard_balanced.csv"
    PROCESSED_DATA_PATH = "../data/processed/"
    MODEL_PATH = "../models/best_model.pkl"
    SCALER_PATH = "../artifacts/scaler.pkl"
    PREDICTION_INPUT_PATH = "../data/raw/new_data.csv"
    PREDICTION_OUTPUT_PATH = "../data/predictions/predictions.csv"

    if args.task == "load_data":
        # Cargar y balancear datos
        print("\n=== Ejecutando: Cargar y Balancear Datos ===")
        data = load_and_balance_data(RAW_DATA_PATH, BALANCED_DATA_PATH)
        print(f"Datos balanceados guardados en: {BALANCED_DATA_PATH}")

    elif args.task == "create_features":
        # Procesar y preseleccionar características
        print("\n=== Ejecutando: Procesar Datos ===")
        data = pd.read_csv(BALANCED_DATA_PATH)
        data_processed = preprocess_data(data)
        data_selected = select_features(data_processed, target_col="Class")
        
        # Dividir en entrenamiento y prueba
        X = data_selected.drop(columns=["Class"])
        y = data_selected["Class"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalar datos y guardar el escalador
        X_train_scaled, X_test_scaled = scale_and_save_scaler(X_train, X_test, SCALER_PATH)
        
        # Guardar conjuntos procesados
        X_train_scaled.to_csv(PROCESSED_DATA_PATH + "X_train.csv", index=False)
        y_train.to_csv(PROCESSED_DATA_PATH + "y_train.csv", index=False)
        X_test_scaled.to_csv(PROCESSED_DATA_PATH + "X_test.csv", index=False)
        y_test.to_csv(PROCESSED_DATA_PATH + "y_test.csv", index=False)
        print(f"Datos procesados guardados en: {PROCESSED_DATA_PATH}")

    elif args.task == "create_models":
        # Entrenar y evaluar modelos
        print("\n=== Ejecutando: Entrenamiento de Modelos ===")
        
        # Cargar datos procesados
        X_train = pd.read_csv(PROCESSED_DATA_PATH + "X_train.csv")
        y_train = pd.read_csv(PROCESSED_DATA_PATH + "y_train.csv")
        X_test = pd.read_csv(PROCESSED_DATA_PATH + "X_test.csv")
        y_test = pd.read_csv(PROCESSED_DATA_PATH + "y_test.csv")

        # Entrenar y evaluar
        best_model, best_model_name, models_with_params = train_and_evaluate_models(X_train, y_train, X_test, y_test)
        best_model = grid_search_best_model(X_train, y_train, best_model_name, models_with_params)
        save_best_model(best_model, MODEL_PATH)
        print(f"Modelo guardado en: {MODEL_PATH}")

    elif args.task == "prediction_pipeline":
        # Realizar predicciones
        print("\n=== Ejecutando: Pipeline de Predicción ===")
        
        # Cargar modelo y datos
        model = load_model(MODEL_PATH)
        data = pd.read_csv(PREDICTION_INPUT_PATH)

        # Preprocesar datos
        data_processed = preprocess_data(data)
        scaler = joblib.load(SCALER_PATH)
        numeric_cols = data_processed.select_dtypes(include=["number"]).columns
        data_processed[numeric_cols] = scaler.transform(data_processed[numeric_cols])

        # Generar predicciones
        predictions, probabilities = predict(model, data_processed)

        # Guardar predicciones
        save_predictions(predictions, probabilities, PREDICTION_OUTPUT_PATH)
        print(f"Predicciones guardadas en: {PREDICTION_OUTPUT_PATH}")

    else:
        print("Tarea no reconocida. Use --help para más información.")
        sys.exit(1)


if __name__ == "__main__":
    main()
