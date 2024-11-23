"""
Script para cargar datos desde un archivo CSV.

Este módulo contiene funciones para cargar datos en un DataFrame de pandas
y validar su estructura inicial.

Funciones:
    - load_data: Carga los datos desde un archivo CSV.
"""

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Carga los datos desde un archivo CSV y los devuelve como un DataFrame.

    Args:
        file_path (str): Ruta del archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo no contiene datos válidos.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Datos cargados correctamente desde {file_path}.")
        print(f"Dimensiones del dataset: {data.shape}")
        return data
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo en la ruta {file_path}.")
        raise e
    except ValueError as e:
        print(f"Error: El archivo no contiene datos válidos en {file_path}.")
        raise e