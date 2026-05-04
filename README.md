# Sinkhole Susceptibility Prediction Model 🕳️

Este trabajo es parte del proyecto SIRIMA, como parte de la investigacion del Research Fund for Coal and Steel (RFCS) que es un fondo de la European Union

Este repositorio contiene el pipeline de Machine Learning y el motor de inferencia desarrollados para predecir la susceptibilidad de formación de sinkholes en zonas mineras de carbon. 

El proyecto combina técnicas de **Procesamiento de Lenguaje Natural (NLP)** para interpretar texto geológico libre y un modelo **Random Forest** para la clasificación del riesgo.

## 📁 Estructura del Repositorio

Para garantizar la trazabilidad y la separación entre los entornos de desarrollo y producción, el código está estructurado de la siguiente manera:

*   **`scripts/`**: Contiene `sinkholes_prediction.ipynb`, el cuaderno principal donde se realizó la limpieza de datos, el Análisis Exploratorio de Datos (EDA) y la imputación inicial de variables objetivo mediante K-Means.
*   **`entrenar_y_empaquetar.py`**: Script orquestador de MLOps. Toma los datos procesados, entrena el pipeline de NLP (TF-IDF) + Random Forest, calcula los umbrales de Jenks y exporta el artefacto final.
*   **`models/`**: Directorio de salida que almacena el modelo serializado (`pipeline_susceptibilidad_NLP_randomforest.pkl`) listo para producción.
*   **`motor_sirima_randomforest.py`**: **Motor de Inferencia**. Es el script de producción que carga el `.pkl` y procesa las peticiones (inputs) para devolver el nivel de riesgo. Está blindado contra errores físicos y de formato.
*   **`test.py`**: Script de pruebas de humo (*smoke test*) para validar los diferentes flujos del motor (NLP, manual y manejo de errores).

> **Nota sobre los datos:** Por motivos de confidencialidad, la carpeta `data/` (que contiene los archivos crudos y .parquet) está excluida de este repositorio mediante `.gitignore`.

## ¿Cómo reproducir el entorno (Local)?

Si deseas ejecutar el código fuente o reentrenar el modelo, sigue estos pasos:

1. Clona este repositorio.
2. Crea un entorno virtual (Conda o venv) con Python 3.9+.
3. Instala las dependencias estrictas de producción:
   ```bash
   pip install -r requirements.txt