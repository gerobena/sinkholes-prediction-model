import os
import pickle
import numpy as np
import pandas as pd
import jenkspy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

print("Iniciando pipeline de empaquetamiento...")

# ==============================================================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==============================================================================
ruta_datos = 'data/processed/'
ruta_empaquetado = 'models/'
os.makedirs(ruta_empaquetado, exist_ok=True)

df_ml = pd.read_parquet(ruta_datos + 'df_ml.parquet')
modelo_granular = pd.read_parquet(ruta_datos + 'modelo_granular.parquet')

# ==============================================================================
# 2. VECTORIZACIÓN (NLP)
# ==============================================================================
print("1/4 Vectorizando descripciones geológicas...")
# Limpieza básica por si no se guardó en el parquet
df_ml['text_for_ml'] = df_ml['geology_of_the_overburden'].astype(str).str.lower().str.strip()

# Inicializamos el vectorizador
vectorizer = TfidfVectorizer(stop_words='english', min_df=4)

# fit_transform hace dos cosas: "aprende" el vocabulario y crea la matriz numérica
X_text = vectorizer.fit_transform(df_ml['text_for_ml'])

# ==============================================================================
# 3. ENTRENAMIENTO DEL MODELO (RANDOM FOREST)
# ==============================================================================
print("2/4 Entrenando clasificador Random Forest...")
# Variable objetivo (la que ya imputaste previamente en el notebook con K-Means)
y_train_res = df_ml['overburden_resistance_numeric'].astype(int)

# Entrenamos el modelo
rf_classifier = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
rf_classifier.fit(X_text, y_train_res)

# ==============================================================================
# 4. PARÁMETROS DE NORMALIZACIÓN Y JENKS
# ==============================================================================
print("3/4 Calculando parámetros de normalización y umbrales...")
# Profundidad (Usando la variable limpia)
log_depth = np.log1p(modelo_granular['depth_drift_clean'].dropna())
log_min_depth, log_max_depth = log_depth.min(), log_depth.max()

# Espesor
thick_min = modelo_granular['working_thickness_numeric'].min()
thick_max = modelo_granular['working_thickness_numeric'].max()

# Umbrales Jenks
scores_finales = modelo_granular['Susceptibility_Score_Expert'].dropna().values
master_breaks = jenkspy.jenks_breaks(scores_finales, n_classes=3)

# ==============================================================================
# 5. DICCIONARIO MAESTRO Y EXPORTACIÓN
# ==============================================================================
print("4/4 Empaquetando y exportando artefacto final...")
pipeline_supervisado = {
    'version': '2.0_NLP_Supervised',
    
    # --- MÓDULO GEOLÓGICO SUPERVISADO ---
    'nlp_vectorizer': vectorizer,
    'nlp_classifier': rf_classifier,
    
    # --- MÓDULO DE SUSCEPTIBILIDAD ---
    # Pesos holisticos (Conocimiento Experto)
    'pesos': {'alpha': 0.4,   # Profundidad
              'beta': 0.3,    # Espesor de capa
              'gamma': 0.3},  # Resistencia del material
              
    # Limites para la variable resistencia de material
    'mapeo_res_a_peso': {1: 0.70, 
                         2: 0.35, 
                         3: 0.10},
    
    # Parametros para profundidad y espesor
    'parametros_normalizacion': {
        'depth_log_min': log_min_depth,
        'depth_log_max': log_max_depth,
        'thick_min': thick_min, 
        'thick_max': thick_max
    },
    
    # Umbrales definidos por Jenks Break
    'umbrales_jenks': {'L1': master_breaks[1], 'L2': master_breaks[2]}
}

ruta_archivo_final = os.path.join(ruta_empaquetado, 'pipeline_susceptibilidad_NLP_randomforest.pkl')

with open(ruta_archivo_final, 'wb') as f:
    pickle.dump(pipeline_supervisado, f)

print(f"✅ ¡Éxito! Pipeline V2 exportado en: {ruta_archivo_final}")