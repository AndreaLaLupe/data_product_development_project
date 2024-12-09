"""
Script para cargar y balancear datos desde un archivo CSV.

Este módulo contiene funciones para cargar datos en un DataFrame de pandas,
inspeccionar su estructura inicial y balancear las clases en una proporción 2:1.

Funciones:
    - load_and_balance_data: Carga los datos desde un archivo CSV, valida su estructura
      y balancea las clases en una proporción 2:1.
"""

import pandas as pd


def load_and_balance_data(file_path: str, balanced_file_path: str) -> pd.DataFrame:
    """
    Carga los datos desde un archivo CSV, valida su estructura inicial y balancea las clases en una proporción 2:1.

    Args:
        file_path (str): Ruta del archivo CSV.
        balanced_file_path (str): Ruta donde se guardará el dataset balanceado.

    Returns:
        pd.DataFrame: DataFrame balanceado.
    """
    try:
        # Cargar los datos
        data = pd.read_csv(file_path)
        print(f"Datos cargados correctamente desde {file_path}.")
        print(f"Dimensiones del dataset: {data.shape}")

        # Inspección inicial
        print("\nPrimeras filas del dataset:")
        print(data.head())
        print("\nInformación del dataset:")
        data.info()
        print("\nEstadísticas descriptivas:")
        print(data.describe())
        print(f"\nValores nulos por columna:\n{data.isnull().sum()}")

        # Verificar la presencia de la columna 'Class'
        if 'Class' not in data.columns:
            raise KeyError("La columna 'Class' es obligatoria para este flujo.")

        # Separar las clases
        data_positiva = data[data['Class'] == 1]
        data_negativa = data[data['Class'] == 0]

        print(f"\nTamaño original de la clase positiva (1): {data_positiva.shape[0]}")
        print(f"Tamaño original de la clase negativa (0): {data_negativa.shape[0]}")

        # Ajustar el tamaño de la clase negativa a 2 veces la clase positiva
        n_samples_negativa = 2 * data_positiva.shape[0]
        data_negativa_balanceada = data_negativa.sample(n=n_samples_negativa, random_state=2024, replace=False)

        # Combinar datos balanceados
        data_balanceada = pd.concat([data_positiva, data_negativa_balanceada])
        data_balanceada = data_balanceada.sample(frac=1, random_state=2024).reset_index(drop=True)  # Mezclar aleatoriamente

        print(f"\nTamaño final del dataset balanceado: {data_balanceada.shape}")
        print("\nDistribución de la variable objetivo (después del balanceo):")
        print(data_balanceada['Class'].value_counts(normalize=True))

        # Guardar el dataset balanceado
        data_balanceada.to_csv(balanced_file_path, index=False)
        print(f"\nEl dataset balanceado ha sido guardado en: {balanced_file_path}")

        return data_balanceada

    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo en la ruta {file_path}.")
        raise e
    except Exception as e:
        print(f"Error: {e}")
        raise e
