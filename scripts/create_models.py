"""
Script para la creación, evaluación y ajuste de modelos.

Este módulo contiene funciones para:
- Entrenar diferentes modelos.
- Evaluar su rendimiento.
- Ajustar hiperparámetros utilizando GridSearchCV.
- Seleccionar el mejor modelo basado en métricas.
- Guardar el mejor modelo como artefacto.

Funciones:
    - train_and_evaluate_models: Entrena y evalúa modelos con sus hiperparámetros.
    - grid_search_best_model: Realiza el ajuste de hiperparámetros para el mejor modelo.
    - save_best_model: Guarda el mejor modelo como artefacto.
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

def load_data(file_path: str) -> tuple:
    """
    Carga los datos preprocesados desde archivos CSV.

    Args:
        file_path (str): Ruta del archivo CSV.

    Returns:
        tuple: X_train, y_train, X_test, y_test
    """
    X_train = pd.read_csv(file_path + 'X_train.csv')
    y_train = pd.read_csv(file_path + 'y_train.csv')
    X_test = pd.read_csv(file_path + 'X_test.csv')
    y_test = pd.read_csv(file_path + 'y_test.csv')

    print(f"Tamaño de los datos de entrenamiento: {X_train.shape}")
    print(f"Tamaño de los datos de prueba: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Entrena varios modelos, los evalúa y selecciona el mejor basado en F1-Score.

    Args:
        X_train (pd.DataFrame): Datos de entrenamiento.
        y_train (pd.Series): Etiquetas de entrenamiento.
        X_test (pd.DataFrame): Datos de prueba.
        y_test (pd.Series): Etiquetas de prueba.

    Returns:
        tuple: El mejor modelo y su nombre.
    """
    models_with_params = {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"], "solver": ["liblinear"]}
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {"max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]}
        },
        "Support Vector Machine": {
            "model": SVC(random_state=42, probability=True),
            "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
            "params": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10], "learning_rate": [0.01, 0.1, 0.2]}
        }
    }

    results = []
    best_model = None
    best_model_name = None
    best_score = 0

    for model_name, model_details in models_with_params.items():
        print(f"\nEntrenando y evaluando: {model_name}")
        model = model_details["model"]
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Calcular métricas
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        # Guardar resultados
        results.append({
            "Modelo": model_name,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        })

        # Seleccionar el mejor modelo dinámicamente
        if f1 > best_score:
            best_model = model
            best_model_name = model_name
            best_score = f1

    # Mostrar resultados
    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    print("\nResultados finales:")
    print(results_df)

    print(f"\nEl mejor modelo es: {best_model_name}")
    return best_model, best_model_name, models_with_params


def grid_search_best_model(X_train, y_train, best_model_name, models_with_params):
    """
    Realiza el ajuste de hiperparámetros para el mejor modelo utilizando GridSearchCV.

    Args:
        X_train (pd.DataFrame): Datos de entrenamiento.
        y_train (pd.Series): Etiquetas de entrenamiento.
        best_model_name (str): Nombre del mejor modelo.
        models_with_params (dict): Diccionario de modelos y sus parámetros.

    Returns:
        object: El mejor modelo ajustado.
    """
    print("\nAjuste de hiperparámetros para el mejor modelo...")
    best_model_params = models_with_params[best_model_name]["params"]
    grid_search = GridSearchCV(
        models_with_params[best_model_name]["model"],
        best_model_params,
        scoring="f1",
        cv=3,
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train.values.ravel())
    best_model = grid_search.best_estimator_

    print("\nMejores parámetros encontrados:")
    print(grid_search.best_params_)

    return best_model


def save_best_model(best_model, model_path):
    """
    Guarda el mejor modelo en un archivo para su uso posterior.

    Args:
        best_model (object): El mejor modelo ajustado.
        model_path (str): Ruta donde se guardará el modelo.
    """
    joblib.dump(best_model, model_path)
    print(f"\nEl mejor modelo ha sido guardado en {model_path}")


if __name__ == "__main__":
    """
    Bloque principal para ejecutar el flujo completo de entrenamiento, evaluación y ajuste de modelos.
    """
    # Cargar los datos procesados
    FILE_PATH = "../data/processed/"
    X_train, y_train, X_test, y_test = load_data(FILE_PATH)

    # Entrenar y evaluar los modelos
    best_model, best_model_name, models_with_params = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    # Ajuste de hiperparámetros para el mejor modelo
    best_model = grid_search_best_model(X_train, y_train, best_model_name, models_with_params)

    # Guardar el mejor modelo
    MODEL_PATH = "../models/best_model.pkl"
    save_best_model(best_model, MODEL_PATH)
