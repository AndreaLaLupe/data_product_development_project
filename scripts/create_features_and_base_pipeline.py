"""
Script para crear un pipeline dinámico, ajustar los datos y generar conjuntos procesados.

Este script utiliza un archivo de configuración para definir los pasos del pipeline,
incluye validación de columnas y genera conjuntos de datos procesados.

Requisitos:
- Archivo de configuración `pipeline.cfg`.
- Dataset balanceado en la ruta `data/interim/creditcard_balanced.csv`.

"""

import os
import pickle
import configparser
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder, CountFrequencyEncoder
from feature_engine.selection import DropFeatures


def validate_columns(data, config, section, key):
    """
    Valida que las columnas especificadas en la configuración existan en el dataset.
    Maneja casos en los que las secciones están vacías.

    Args:
        data (pd.DataFrame): El dataset a validar.
        config (ConfigParser): Archivo de configuración cargado.
        section (str): Nombre de la sección en el archivo de configuración.
        key (str): Clave de la sección a validar.

    Returns:
        list: Lista de columnas validadas o vacía si no hay columnas especificadas.
    """
    if key not in config[section] or not config[section][key].strip():
        print(f"No se encontraron columnas para {section} -> {key}.")
        return []

    columns = [col.strip() for col in config[section][key].split(',')]
    missing_columns = [col for col in columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Las siguientes columnas no existen en el dataset: {missing_columns}"
        )
    return columns


def load_data(raw_data_path):
    """
    Carga el dataset desde la ruta especificada.

    Args:
        raw_data_path (str): Ruta al archivo CSV.

    Returns:
        pd.DataFrame: Dataset cargado.
    """
    try:
        data = pd.read_csv(raw_data_path)
        print(f"Datos cargados desde {raw_data_path}. Dimensiones: {data.shape}")
        return data
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"El archivo {raw_data_path} no existe. Verifica la ruta."
        ) from exc


def save_pipeline(pipeline, artifacts_path):
    """
    Guarda el pipeline como un archivo pickle.

    Args:
        pipeline (Pipeline): Objeto pipeline a guardar.
        artifacts_path (str): Ruta donde guardar el archivo.
    """
    pipeline_path = os.path.join(artifacts_path, "base_pipeline.pkl")
    try:
        with open(pipeline_path, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"\nPipeline guardado como archivo: {pipeline_path}")
    except Exception as exc:
        raise ValueError(f"Error al guardar el pipeline: {exc}") from exc


def process_and_save_data(pipeline, data, config, processed_data_path):
    """
    Procesa los datos usando el pipeline y los guarda como archivos CSV.

    Args:
        pipeline (Pipeline): Pipeline configurado.
        data (pd.DataFrame): Dataset a procesar.
        config (ConfigParser): Archivo de configuración.
        processed_data_path (str): Ruta para guardar los datos procesados.
    """
    try:
        x = data.drop(columns=[config['GENERAL']['TARGET']])
        y = data[config['GENERAL']['TARGET']]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline.fit(x_train, y_train)

        x_train_processed = pipeline.transform(x_train)
        x_test_processed = pipeline.transform(x_test)

        pd.DataFrame(x_train_processed).to_csv(
            os.path.join(processed_data_path, "X_train.csv"), index=False
        )
        y_train.to_csv(os.path.join(processed_data_path, "y_train.csv"), index=False)
        pd.DataFrame(x_test_processed).to_csv(
            os.path.join(processed_data_path, "X_test.csv"), index=False
        )
        y_test.to_csv(os.path.join(processed_data_path, "y_test.csv"), index=False)

        print("\nConjuntos de datos procesados guardados correctamente.")
    except Exception as exc:
        raise ValueError(f"Error al procesar los datos: {exc}") from exc


def main():
    """
    Función principal del script.
    """
    # Configuración inicial
    project_path = os.getcwd()
    raw_data_path = os.path.join(project_path, "..", "data", "interim", "creditcard_balanced.csv")
    artifacts_path = os.path.join(project_path, "..", "artifacts")
    processed_data_path = os.path.join(project_path, "..", "data", "processed")
    config_path = os.path.join(project_path, "..", "pipeline.cfg")

    os.makedirs(artifacts_path, exist_ok=True)
    os.makedirs(processed_data_path, exist_ok=True)

    data = load_data(raw_data_path)

    # Leer configuración
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"El archivo de configuración {config_path} no existe."
        )
    config.read(config_path)

    # Validar columnas
    vars_to_drop = validate_columns(data, config, 'GENERAL', 'VARS_TO_DROP')
    vars_to_impute_continues = validate_columns(data, config, 'CONTINUES', 'VARS_TO_IMPUTE')
    vars_to_impute_categorical = validate_columns(data, config, 'CATEGORICAL', 'VARS_TO_IMPUTE')
    ohe_vars = validate_columns(data, config, 'CATEGORICAL', 'OHE_VARS')
    freq_enc_vars = validate_columns(data, config, 'CATEGORICAL', 'FREQUENCY_ENC_VARS')

    # Crear pipeline dinámico
    pipeline_steps = []

    if vars_to_drop:
        pipeline_steps.append(('delete_features', DropFeatures(features_to_drop=vars_to_drop)))

    if vars_to_impute_continues:
        pipeline_steps.append(('mean_imputer', MeanMedianImputer(
            imputation_method='mean', variables=vars_to_impute_continues
        )))

    if vars_to_impute_categorical:
        pipeline_steps.append(('categorical_imputer', CategoricalImputer(
            imputation_method='frequent', variables=vars_to_impute_categorical
        )))

    if ohe_vars:
        pipeline_steps.append(
            ('one_hot_encoder', OneHotEncoder(variables=ohe_vars, drop_last=True))
        )

    if freq_enc_vars:
        pipeline_steps.append(('frequency_encoder', CountFrequencyEncoder(
            encoding_method='count', variables=freq_enc_vars
        )))

    pipeline_steps.append(('scaling', StandardScaler()))

    pipeline = Pipeline(pipeline_steps)
    print("\nPipeline creado dinámicamente con los siguientes pasos:")
    for step_name, step in pipeline.steps:
        print(f"- {step_name}: {step.__class__.__name__}")

    save_pipeline(pipeline, artifacts_path)
    process_and_save_data(pipeline, data, config, processed_data_path)


if __name__ == "__main__":
    main()
