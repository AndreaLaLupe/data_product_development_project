"""
Script para cargar datos, inspeccionarlos y generar un archivo de configuración basado 
en el análisis del dataset.
Requisitos:
- Cumplimiento del estándar PEP8.
"""

import os
import configparser
import numpy as np
import pandas as pd

# Configuración inicial
PROJECT_PATH = os.path.abspath(os.path.join(os.getcwd()))
FILE_PATH = os.path.join(PROJECT_PATH,"data", "interim", "creditcard_balanced.csv")
CONFIG = os.path.join(PROJECT_PATH, "config")
CONFIG_PATH = os.path.join(CONFIG, "pipeline.cfg")
os.makedirs(CONFIG, exist_ok=True)

def cargar_datos(file_path):
    """
    Cargar datos desde un archivo CSV.

    Args:
        file_path (str): Ruta del archivo CSV.

    Returns:
        pd.DataFrame: Dataset cargado.
    """
    data = pd.read_csv(file_path)
    print("Dimensiones del dataset:", data.shape)
    print("Primeras filas del dataset:")
    print(data.head())

    if 'Class' in data.columns:
        print("\nDistribución de la variable objetivo (Class):")
        print(data['Class'].value_counts(normalize=True))
    else:
        print("\nLa columna 'Class' no está presente en los datos.")

    return data


def manejar_valores_faltantes(data):
    """
    Manejar valores faltantes en el dataset imputándolos con la media.

    Args:
        data (pd.DataFrame): Dataset procesado.

    Returns:
        pd.DataFrame: Dataset con valores faltantes imputados.
    """
    print("\nValores faltantes por columna:")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0])

    if missing_values.sum() > 0:
        data.fillna(data.mean(), inplace=True)
        print("\nSe imputaron valores faltantes con la media.")
    else:
        print("No se encontraron valores faltantes.")

    return data


def identificar_tipos_variables(data):
    """
    Identificar columnas categóricas y numéricas en el dataset.

    Args:
        data (pd.DataFrame): Dataset procesado.

    Returns:
        tuple: Listas de columnas categóricas y numéricas.
    """
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    print("\nColumnas categóricas identificadas:", categorical_cols)
    print("\nColumnas numéricas identificadas:", numeric_cols)

    return categorical_cols, numeric_cols


def generar_configuracion(data, target_column, output_path, categorical_cols, numeric_cols):
    """
    Genera un archivo de configuración basado en el análisis del dataset.

    Args:
        data (pd.DataFrame): Dataset procesado.
        target_column (str): Nombre de la columna objetivo.
        output_path (str): Ruta donde se guardará el archivo de configuración.
        categorical_cols (list): Columnas categóricas.
        numeric_cols (list): Columnas numéricas.
    """
    config = configparser.ConfigParser()

    # [GENERAL]
    redundant_features = []
    correlation_matrix = data.corr()
    for col in correlation_matrix.columns:
        high_corr = correlation_matrix[col][correlation_matrix[col] > 0.8].index.drop(col)
        if len(high_corr) > 0:
            redundant_features.append(col)

    config['GENERAL'] = {
        'VARS_TO_DROP': ', '.join(redundant_features),
        'TARGET': target_column
    }

    # [CONTINUES]
    vars_to_impute_continues = [col for col in numeric_cols if data[col].isnull().sum() > 0]
    config['CONTINUES'] = {
        'VARS_TO_IMPUTE': ', '.join(vars_to_impute_continues)
    }

    # [CATEGORICAL]
    vars_to_impute_categorical = [col for col in categorical_cols if data[col].isnull().sum() > 0]
    ohe_vars = [col for col in categorical_cols if data[col].nunique() <= 10]
    freq_enc_vars = [col for col in categorical_cols if data[col].nunique() > 10]

    config['CATEGORICAL'] = {
        'VARS_TO_IMPUTE': ', '.join(vars_to_impute_categorical),
        'OHE_VARS': ', '.join(ohe_vars),
        'FREQUENCY_ENC_VARS': ', '.join(freq_enc_vars)
    }

    # Validar y eliminar archivo existente
    if os.path.exists(output_path):
        print(f"El archivo {output_path} ya existe. Eliminándolo...")
        os.remove(output_path)

    with open(output_path, 'w', encoding='utf-8') as configfile:
        config.write(configfile)

    print(f"Archivo de configuración generado en: {output_path}")


def main():
    """
    Función principal para cargar datos, procesarlos y generar un archivo de configuración.
    """
    data = cargar_datos(FILE_PATH)
    data = manejar_valores_faltantes(data)
    categorical_cols, numeric_cols = identificar_tipos_variables(data)

    generar_configuracion(
        data=data,
        target_column='Class',
        output_path=CONFIG_PATH,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols
    )


if __name__ == "__main__":
    main()
