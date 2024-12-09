# Librerías
import os
import pandas as pd
import pickle
import mlflow
from datetime import datetime

# Configuración inicial
#PROJECT_PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))
PROJECT_PATH = os.getcwd()
DATA_PROCESSED_PATH = os.path.join(PROJECT_PATH, "data", "processed")
PREDICTIONS_PATH = os.path.join(PROJECT_PATH, "data", "predictions")
ARTIFACTS_PATH = os.path.join(PROJECT_PATH, "artifacts")
PIPELINE_PATH = os.path.join(ARTIFACTS_PATH, "best_model_pipeline.pkl")

# Crear carpeta de predicciones si no existe
os.makedirs(PREDICTIONS_PATH, exist_ok=True)

# Cargar pipeline
def cargar_pipeline(pipeline_path):
    """
    Carga el pipeline entrenado desde un archivo .pkl.
    
    Args:
        pipeline_path (str): Ruta del archivo del pipeline entrenado.
    
    Returns:
        Pipeline: Objeto pipeline cargado.
    """
    try:
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)
        print("Pipeline cargado correctamente.")
        return pipeline
    except FileNotFoundError:
        raise FileNotFoundError(f"El archivo del pipeline no existe en: {pipeline_path}")

# Cargar datos
def cargar_datos_prueba(data_path):
    """
    Carga los datos de prueba desde la ruta especificada.
    
    Args:
        data_path (str): Ruta de los datos procesados.
    
    Returns:
        tuple: DataFrames de X_test e y_test.
    """
    try:
        X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"))
        print(f"Datos de prueba cargados: X_test {X_test.shape}, y_test {y_test.shape}")
        return X_test, y_test
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error al cargar los datos de prueba: {e}")

# Guardar las predicciones
def guardar_predicciones(predictions, true_labels, output_path):
    """
    Guarda las predicciones en un archivo CSV.
    
    Args:
        predictions (array): Predicciones del modelo.
        true_labels (array): Etiquetas verdaderas.
        output_path (str): Ruta donde guardar el archivo CSV.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = f"predictions_{timestamp}.csv"
    output_file = os.path.join(output_path, file_name)
    
    predictions_df = pd.DataFrame({
        "True_Label": true_labels,
        "Predicted_Label": predictions
    })
    
    predictions_df.to_csv(output_file, index=False)
    print(f"Predicciones guardadas en: {output_file}")
    return output_file

# Registrar en MLFlow
def registrar_predicciones_mlflow(file_path):
    """
    Registra las predicciones en MLflow.
    
    Args:
        file_path (str): Ruta del archivo de predicciones.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000") 
    mlflow.set_experiment("Predicciones_Modelo")
    
    with mlflow.start_run():
        mlflow.log_artifact(file_path, artifact_path="predictions")
        print("Predicciones registradas en MLflow.")

# Función principal
def main():
    """
    Función principal del script.
    """
    print("Cargando el pipeline entrenado...")
    pipeline = cargar_pipeline(PIPELINE_PATH)
    
    print("\nCargando datos de prueba...")
    X_test, y_test = cargar_datos_prueba(DATA_PROCESSED_PATH)
    
    print("\nRealizando predicciones...")
    y_pred = pipeline.predict(X_test)
    
    print("\nGuardando predicciones en archivo...")
    output_file = guardar_predicciones(y_pred, y_test.values.ravel(), PREDICTIONS_PATH)

    print("\nRegistrando predicciones en MLflow...")
    registrar_predicciones_mlflow(output_file)
    
    print("\nProceso completado con éxito.")

if __name__ == "__main__":
    main()
