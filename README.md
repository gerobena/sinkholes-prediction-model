# Sinkhole Susceptibility Prediction Model 🕳️

This work is part of the **SIRIMA project**, conducted as research for the **Research Fund for Coal and Steel (RFCS)**, which is a fund of the **European Union**.

This repository contains the Operational Tool for sinkhole hazard assesment. The first part implies a Machine Learning pipeline and the inference engine developed to predict sinkhole susceptibility in coal mining areas. The project combines **Natural Language Processing (NLP)** techniques to interpret raw geological text and a **Random Forest** model for hazard classification. The second part is related to the manitude or energy learverage for the sinkhole ocurrence (in progress)

> **🚧 Project Status:** At this point, the **first part** of the operational tool for the SIRIMA project has been developed. The second part, related to **sinkhole magnitude**, is currently pending. This subsequent phase is required to ultimately calculate the final **hazard index**.

## 📁 Repository Structure

To ensure traceability and the separation between development and production environments, the code is structured as follows:

*   **`scripts/`**: Contains `sinkholes_prediction.ipynb`, the main notebook where data cleaning, Exploratory Data Analysis (EDA), and initial target variable imputation were performed.
*   **`train_and_package.py`**: Orchestrator script. It takes the processed data, trains the NLP pipeline (TF-IDF) + Random Forest, calculates Jenks natural breaks, and exports the final artifact.
*   **`models/`**: Output directory storing the serialized model (`pipeline_susceptibilidad_NLP_randomforest.pkl`) ready for production.
*   **`motor_sirima_randomforest.py`**: **Inference Engine**. This is the production script that loads the `.pkl` file and processes requests (inputs) to return the susceptibility level. It is shielded against physical and formatting errors.
*   **`test.py`**: Test script to validate different engine flows (NLP, manual, and error handling). This script is intended for the end-user to test the model.

> **Note on Data:** For confidentiality reasons, the `data/` folder (containing raw and .parquet files) is excluded from this repository via `.gitignore`.

## ⚙️ How to reproduce the environment (Local)?

If you wish to run the source code or retrain the model, follow these steps:

1.  Clone this repository.
2.  Create a virtual environment (Conda or venv) with Python 3.9+.
3.  Install strict production dependencies:
    ```bash
    pip install -r requirements.txt