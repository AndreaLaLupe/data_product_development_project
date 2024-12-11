"""
Este script entrena modelos, ajusta hiperparámetros,
selecciona el mejor modelo y configura un pipeline
con el modelo ganador.

Cumple con el estándar PEP8 y utiliza Pylint
para asegurar calidad del código.
"""

import os
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Configuración inicial
PROJECT_PATH = os.path.abspath(os.path.join(os.getcwd()))
DATA_PROCESSED_PATH = os.path.join(PROJECT_PATH, "data", "processed")
ARTIFACTS_PATH = os.path.join(PROJECT_PATH, "artifacts")
PIPELINE_PATH = os.path.join(ARTIFACTS_PATH, "base_pipeline.pkl")
FINAL_PIPELINE_PATH = os.path.join(ARTIFACTS_PATH, "best_model_pipeline.pkl")
RESULTS_FILE_PATH = os.path.join(ARTIFACTS_PATH, "model_results.json")

os.makedirs(ARTIFACTS_PATH, exist_ok=True)


def load_data():
    """
    Carga los datos procesados desde la ruta definida.

    Returns:
        tuple: DataFrames de entrenamiento y prueba (x_train, y_train, x_test, y_test).
    """
    try:
        print("\nCargando datos procesados...")
        x_train = pd.read_csv(os.path.join(DATA_PROCESSED_PATH, "X_train.csv"), encoding="utf-8")
        y_train = pd.read_csv(os.path.join(DATA_PROCESSED_PATH, "y_train.csv"), encoding="utf-8")
        x_test = pd.read_csv(os.path.join(DATA_PROCESSED_PATH, "X_test.csv"), encoding="utf-8")
        y_test = pd.read_csv(os.path.join(DATA_PROCESSED_PATH, "y_test.csv"), encoding="utf-8")
        return x_train, y_train, x_test, y_test
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Error al cargar los datos procesados: {error}") from error


def define_models():
    """
    Define los modelos disponibles para entrenar.

    Returns:
        dict: Diccionario con los modelos a entrenar.
    """
    return {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVC": SVC(random_state=42, probability=True),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss")
    }


def get_hyperparameter_grid(model_name):
    """
    Devuelve el grid de hiperparámetros para el modelo especificado.

    Args:
        model_name (str): Nombre del modelo.

    Returns:
        dict: Grid de hiperparámetros.
    """
    param_grids = {
        "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
        "Decision Tree": {"max_depth": [5, 10, 20, None]},
        "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
        "SVC": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        "XGBoost": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7]}
    }
    return param_grids.get(model_name, {})


def train_model_with_hyperparameters(model_name, model, param_grid, x_train, y_train):
    """
    Entrena el modelo y ajusta los hiperparámetros si se define un grid.

    Args:
        model_name (str): Nombre del modelo.
        model: Modelo a entrenar.
        param_grid (dict): Grid de hiperparámetros.
        x_train (DataFrame): Datos de entrenamiento.
        y_train (DataFrame): Etiquetas de entrenamiento.

    Returns:
        tuple: Modelo entrenado y mejores hiperparámetros.
    """
    if param_grid:
        print(f"Ajustando hiperparámetros para {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
        grid_search.fit(x_train, y_train.values.ravel())
        return grid_search.best_estimator_, grid_search.best_params_
    model.fit(x_train, y_train.values.ravel())
    return model, None


def train_and_tune_models(x_train, y_train, x_test, y_test, models_with_params):
    """
    Entrena y ajusta todos los modelos especificados.

    Args:
        x_train (DataFrame): Datos de entrenamiento.
        y_train (DataFrame): Etiquetas de entrenamiento.
        x_test (DataFrame): Datos de prueba.
        y_test (DataFrame): Etiquetas de prueba.
        models_with_params (dict): Diccionario de modelos a entrenar.

    Returns:
        dict: Resultados de los modelos entrenados.
    """
    results = {}
    print("\nEntrenando modelos y evaluando...")
    for model_name, model in models_with_params.items():
        print(f"\nEntrenando modelo: {model_name}")
        param_grid = get_hyperparameter_grid(model_name)
        best_model, best_params = train_model_with_hyperparameters(
            model_name, model, param_grid, x_train, y_train
        )

        y_pred = best_model.predict(x_test)
        f1 = f1_score(y_test, y_pred)
        print(f"F1-Score para {model_name}: {f1}")

        model_file = os.path.join(
            ARTIFACTS_PATH, f"{model_name.replace(' ', '_').lower()}.pkl"
        )
        with open(model_file, "wb") as file:
            pickle.dump(best_model, file)

        results[model_name] = {
            "f1_score": f1,
            "hyperparameters": best_params,
            "model_path": model_file
        }
    return results


def save_results(results):
    """
    Guarda los resultados de los modelos en un archivo JSON.

    Args:
        results (dict): Resultados de los modelos.
    """
    with open(RESULTS_FILE_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file)
    print(f"\nResultados guardados en {RESULTS_FILE_PATH}")


def load_pipeline():
    """
    Carga el pipeline base desde el archivo.

    Returns:
        Pipeline: Objeto pipeline cargado.
    """
    print("\nCargando pipeline base...")
    with open(PIPELINE_PATH, "rb") as file:
        return pickle.load(file)


def update_pipeline_with_model(pipeline, x_train, best_model):
    """
    Actualiza el pipeline con el modelo ganador.

    Args:
        pipeline (Pipeline): Pipeline base.
        x_train (DataFrame): Datos de entrenamiento.
        best_model: Modelo ganador.

    Returns:
        Pipeline: Pipeline actualizado.
    """
    def validate_pipeline_columns(pipeline_obj, x_data):
        updated_steps = []
        for name, step in pipeline_obj.steps:
            if hasattr(step, "features_to_drop"):
                valid_features = [col for col in step.features_to_drop if col in x_data.columns]
                if valid_features:
                    step.features_to_drop = valid_features
                    updated_steps.append((name, step))
                else:
                    print(f"El paso '{name}' fue eliminado porque no tiene columnas válidas.")
            else:
                updated_steps.append((name, step))
        return Pipeline(updated_steps)

    pipeline = validate_pipeline_columns(pipeline, x_train)
    pipeline.steps.append(("best_model", best_model))
    print("Modelo ganador agregado al pipeline.")
    return pipeline


def save_pipeline(pipeline):
    """
    Guarda el pipeline ajustado en un archivo.

    Args:
        pipeline (Pipeline): Pipeline ajustado.
    """
    print(f"\nGuardando el pipeline ajustado en: {FINAL_PIPELINE_PATH}")
    with open(FINAL_PIPELINE_PATH, "wb") as file:
        pickle.dump(pipeline, file)


def evaluate_pipeline(pipeline, x_test, y_test):
    """
    Evalúa el pipeline ajustado.

    Args:
        pipeline (Pipeline): Pipeline ajustado.
        x_test (DataFrame): Datos de prueba.
        y_test (DataFrame): Etiquetas de prueba.
    """
    print("\nEvaluando el pipeline ajustado...")
    y_pred = pipeline.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    print(f"\nF1-Score del pipeline ajustado: {f1}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))


def main():
    """
    Función principal para entrenar, seleccionar el mejor modelo,
    configurar el pipeline y guardar el modelo final.
    """
    x_train, y_train, x_test, y_test = load_data()
    models_with_params = define_models()
    results = train_and_tune_models(x_train, y_train, x_test, y_test, models_with_params)
    save_results(results)

    best_model_name, best_model_path = max(
        results.items(), key=lambda item: item[1]["f1_score"]
    )[0], results[max(results, key=lambda x: results[x]["f1_score"])]["model_path"]

    print(f"Mejor modelo: {best_model_name}")
    with open(best_model_path, "rb") as file:
        best_model = pickle.load(file)

    pipeline = load_pipeline()
    pipeline = update_pipeline_with_model(pipeline, x_train, best_model)
    pipeline.fit(x_train, y_train.values.ravel())
    save_pipeline(pipeline)
    evaluate_pipeline(pipeline, x_test, y_test)


if __name__ == "__main__":
    main()
