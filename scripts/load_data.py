"""
Script para cargar y balancear datos desde un archivo CSV.

Este módulo contiene funciones para cargar datos en un DataFrame de pandas,
inspeccionar su estructura inicial y aplicar SMOTE para balancear las clases.

Funciones:
    - load_and_balance_data: Carga los datos desde un archivo CSV, valida su estructura
      y balancea las clases utilizando SMOTE.
"""

import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors


def load_and_balance_data(file_path: str, balanced_file_path: str) -> pd.DataFrame:
    """
    Carga los datos desde un archivo CSV, valida su estructura inicial y balancea las clases.

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

        # Separar características y variable objetivo
        X = data.drop(columns=['Class'])
        y = data['Class']

        # Verificar la distribución de clases
        print("\nDistribución de la variable objetivo (antes del balanceo):")
        print(data['Class'].value_counts(normalize=True))

        # Aplicar SMOTE para balancear las clases
        print("\nAplicando SMOTE para balancear las clases...")
        knn_estimator = NearestNeighbors(n_jobs=4)  # Configurar n_jobs en el estimador
        smote = SMOTE(random_state=42, k_neighbors=knn_estimator)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Verificar la distribución después de SMOTE
        print("\nDistribución de la variable objetivo (después de aplicar SMOTE):")
        print(Counter(y_resampled))

        # Crear un nuevo DataFrame balanceado
        data_balanced = pd.concat(
            [pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['Class'])],
            axis=1
        )

        # Guardar el dataset balanceado
        data_balanced.to_csv(balanced_file_path, index=False)
        print(f"\nEl dataset balanceado ha sido guardado en: {balanced_file_path}")

        return data_balanced

    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo en la ruta {file_path}.")
        raise e
    except Exception as e:
        print(f"Error: {e}")
        raise e
