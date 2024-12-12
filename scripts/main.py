"""
Script principal para ejecutar las diferentes etapas del proyecto.
Este script permite al usuario seleccionar qué etapa desea ejecutar
a través de un menú interactivo.
"""

import os
import subprocess

# Configuración inicial
PROJECT_PATH = os.getcwd()

# Opciones del menú
MENU_OPTIONS = {
    "1": "Cargar y balancear datos",
    "2": "Generar archivo de configuración",
    "3": "Crear características y pipeline base",
    "4": "Configurar, entrenar y ajustar el pipeline",
    "5": "Pipeline de predicción",
    "0": "Salir"
}


def run_script(script_name):
    """
    Ejecuta un script desde la línea de comandos.

    Args:
        script_name (str): Nombre del script a ejecutar.
    """
    try:
        print(f"\nEjecutando el script: {script_name}...")
        subprocess.run(
            ["python", os.path.join(PROJECT_PATH, "scripts", script_name)],
            check=True,
        )
        print(f"\n{script_name} ejecutado exitosamente.")
    except subprocess.CalledProcessError as error:
        print(f"\nError al ejecutar el script {script_name}: {error}")
    except FileNotFoundError:
        print(f"\nEl script {script_name} no se encuentra en la ruta especificada.")


def main():
    """
    Menú principal para seleccionar y ejecutar scripts del proyecto.
    """
    while True:
        print("\nMenú principal:")
        for key, value in MENU_OPTIONS.items():
            print(f"{key}. {value}")

        option = input("\nSeleccione una opción: ").strip()

        if option == "1":
            run_script("load_data.py")
        elif option == "2":
            run_script("configuration_file.py")
        elif option == "3":
            run_script("create_features_and_base_pipeline.py")
        elif option == "4":
            run_script("configure_and_fit_pipeline.py")
        elif option == "5":
            run_script("predict_pipeline.py")
        elif option == "0":
            print("\nSaliendo del programa...")
            break
        else:
            print("\nOpción no válida. Intente de nuevo.")


if __name__ == "__main__":
    main()
