# Sinkhole Susceptibility Prediction Model 🕳️

This work is part of the **SIRIMA project**, conducted as research for the **Research Fund for Coal and Steel (RFCS)**, which is a fund of the **European Union**.

This repository contains the **Operational Tool** for sinkhole hazard assessment. It encompasses a comprehensive data analytic pipeline, a Machine Learning inference engine, and a web-based user interface developed to predict sinkhole susceptibility and final hazard across historical transnational coal mining areas (France, Germany, and Poland).

> **✅ Project Status:** The operational tool for the SIRIMA project integrates the **Susceptibility Module**, the **Magnitude Module**, and the **Hazard Assessment**, all deployed via an interactive web application.

## 📁 Repository Structure

To ensure traceability and the separation between development and production environments, the code is structured as follows:

*  **`app.py`**: **Streamlit Web Application**. This is the user-friendly interface for deploying the tool. It allows users to input physical/geological parameters and receive visual hazard cards.

*   **`scripts/`**: Contains `sinkholes_prediction.ipynb`, the main notebook where data cleaning, Exploratory Data Analysis (EDA), and initial target variable imputation were performed.

*   **`train_and_package.py`**: Orchestrator script. It takes the processed data, trains the NLP pipeline (TF-IDF) + Random Forest, calculates Jenks natural breaks, and exports the final artifact.

*   **`models/pipeline_sirima_completo.pkl`**: It contains all results from `sinkholes_prediction.ipynb` packeged together.

*   **`motor_sirima_completo.py`**: **Inference Engine**. This is the production script that loads the `.pkl` file and processes requests (inputs) to return the susceptibility level. It is shielded against physical and formatting errors.

*   **`test.py`**: Test script to validate different engine flows (NLP, manual, and error handling). This script is intended for the end-user to test the model.

> **Note on Data:** For confidentiality reasons, the `data/` folder (containing raw and .parquet files) is excluded from this repository via `.gitignore`.

## ⚙️ How to reproduce the environment (Local)?

If you wish to run the source code or retrain the model, follow these steps:

1.  Clone this repository.
2.  Create a virtual environment (Conda or venv) with Python 3.9+.
3.  Install strict production dependencies:
    ```bash
    pip install -r requirements.txt
4.  Run the Streamlit application:
    ```bash
    streamlit run scripts/app.py
    