"""
Script para configurar y entrenar un pipeline de machine learning.
Selecciona el mejor modelo basado en la métrica F1-Score, ajusta
hiperparámetros, y guarda el pipeline ajustado en artefactos.
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def cargar_datos(data_path):
    """
    Cargar datos procesados de entrenamiento y prueba.

    Args:
        data_path (str): Ruta de la carpeta que contiene los datos procesados.

    Returns:
        tuple: DataFrames x_train, y_train, x_test, y_test.
    """
    try:
        x_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"))
        x_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"))

        print(f"Datos cargados: x_train {x_train.shape}, y_train {y_train.shape}")
        return x_train, y_train, x_test, y_test
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Error al cargar los datos procesados: {exc}") from exc


def entrenar_modelos(x_train, y_train, x_test, y_test):
    """
    Entrenar varios modelos y seleccionar el mejor basado en F1-Score.

    Args:
        x_train (pd.DataFrame): Datos de entrenamiento.
        y_train (pd.DataFrame): Etiquetas de entrenamiento.
        x_test (pd.DataFrame): Datos de prueba.
        y_test (pd.DataFrame): Etiquetas de prueba.

    Returns:
        tuple: Modelo ganador, nombre del modelo y mejor F1-Score.
    """
    models_with_params = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVC": SVC(random_state=42, probability=True),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
    }

    best_model = None
    best_model_name = None
    best_f1_score = 0

    print("\nEntrenando modelos y evaluando...")
    for model_name, model in models_with_params.items():
        print(f"\nEntrenando modelo: {model_name}")
        model.fit(x_train, y_train.values.ravel())
        y_pred = model.predict(x_test)
        f1 = f1_score(y_test, y_pred)

        print(f"F1-Score para {model_name}: {f1}")

        if f1 > best_f1_score:
            best_model = model
            best_model_name = model_name
            best_f1_score = f1

    print(f"\nMejor modelo: {best_model_name} con F1-Score: {best_f1_score}")
    return best_model, best_model_name, best_f1_score


def ajustar_hiperparametros(model, model_name, x_train, y_train):
    """
    Ajustar hiperparámetros para Random Forest si es el mejor modelo.

    Args:
        model: Modelo seleccionado.
        model_name (str): Nombre del modelo seleccionado.
        x_train (pd.DataFrame): Datos de entrenamiento.
        y_train (pd.DataFrame): Etiquetas de entrenamiento.

    Returns:
        Modelo ajustado.
    """
    if model_name == "Random Forest":
        print("\nAjustando hiperparámetros para Random Forest...")
        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
        grid_search.fit(x_train, y_train.values.ravel())
        print("Mejores parámetros encontrados:", grid_search.best_params_)
        return grid_search.best_estimator_
    return model


def guardar_modelo(model, artifacts_path):
    """
    Guardar el modelo ajustado por separado.

    Args:
        model: El modelo ajustado.
        artifacts_path (str): Ruta donde guardar el modelo.
    """
    model_path = os.path.join(artifacts_path, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModelo guardado en: {model_path}")


def validar_pipeline(pipeline, x_train):
    """
    Validar columnas esperadas por el pipeline y ajustar pasos inválidos.

    Args:
        pipeline (Pipeline): Pipeline cargado.
        x_train (pd.DataFrame): Datos de entrenamiento.

    Returns:
        Pipeline actualizado.
    """
    updated_steps = []
    for name, step in pipeline.steps:
        if hasattr(step, 'features_to_drop'):
            valid_features = [col for col in step.features_to_drop if col in x_train.columns]
            if valid_features:
                step.features_to_drop = valid_features
                updated_steps.append((name, step))
            else:
                print(f"El paso '{name}' fue eliminado porque no tiene columnas válidas.")
        else:
            updated_steps.append((name, step))
    return Pipeline(updated_steps)


def main():
    """
    Función principal para cargar datos, entrenar el pipeline,
    guardar el modelo ajustado y evaluar su desempeño.
    """
    # Configuración inicial
    project_path = os.getcwd()
    data_processed_path = os.path.join(project_path, "data", "processed")
    artifacts_path = os.path.join(project_path, "artifacts")
    pipeline_path = os.path.join(artifacts_path, "base_pipeline.pkl")
    model_pipeline_path = os.path.join(artifacts_path, "best_model_pipeline.pkl")

    # Cargar datos
    x_train, y_train, x_test, y_test = cargar_datos(data_processed_path)

    # Entrenar modelos y seleccionar el mejor
    best_model, best_model_name, _ = entrenar_modelos(x_train, y_train, x_test, y_test)

    # Ajustar hiperparámetros si es necesario
    best_model = ajustar_hiperparametros(best_model, best_model_name, x_train, y_train)

    # Guardar el mejor modelo
    guardar_modelo(best_model, artifacts_path)

    # Cargar y validar pipeline base
    print("\nCargando pipeline base...")
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    pipeline = validar_pipeline(pipeline, x_train)
    pipeline.steps.append(("best_model", best_model))

    # Ajustar el pipeline completo
    try:
        print("\nAjustando el pipeline completo...")
        pipeline.fit(x_train, y_train.values.ravel())
    except Exception as exc:
        raise ValueError(f"Error al ajustar el pipeline completo: {exc}") from exc

    # Guardar pipeline ajustado
    print("\nGuardando el pipeline ajustado...")
    with open(model_pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)

    # Evaluar pipeline ajustado
    print("\nEvaluando el pipeline ajustado...")
    y_pred = pipeline.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    print(f"\nF1-Score del pipeline ajustado: {f1}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
