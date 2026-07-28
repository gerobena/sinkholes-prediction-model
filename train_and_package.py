import os
import pickle
import numpy as np
import pandas as pd
import jenkspy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

print("Iniciando pipeline de empaquetamiento...")

# ==============================================================================
# 0. PANEL DE CONTROL DE PARÁMETROS
# ==============================================================================
# Pesos del modelo de Susceptibilidad (Deben sumar 1.0)
CONFIG_PESOS = {'alpha': 0.4, 
                'beta': 0.3, 
                'gamma': 0.3
                }

# Multiplicadores de Resistencia (Clase 1=Baja, 2=Media, 3=Alta)
CONFIG_RESISTENCIA = {1: 0.70, 
                      2: 0.35, 
                      3: 0.10
                      }

# Umbrales (Bins) del Modelo de Magnitud
CONFIG_BINS_MAGNITUD = {
    'thick_bins': [-1, 2.0, 3.25, 100],
    'thick_labels': ['1_Delgada (<2.0m)', '2_Media (2.0-3.25m)', '3_Gruesa (>3.25m)'],
    'dip_bins': [-1, 5, 15, 90],
    'dip_labels': ['1_Plano (0-5°)', '2_Suave (5-15°)', '3_Inclinado (>15°)']
}

# ==============================================================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==============================================================================
ruta_datos = 'data/processed/'
ruta_empaquetado = 'models/'
os.makedirs(ruta_empaquetado, exist_ok=True)

# Cargamos los dataset finales
df_ml = pd.read_parquet(ruta_datos + 'df_ml.parquet')
modelo_granular = pd.read_parquet(ruta_datos + 'modelo_granular.parquet')
df_magnitud = pd.read_parquet(ruta_datos + 'modelo_magnitud.parquet')

# ==============================================================================
# 2. VECTORIZACIÓN (NLP)
# ==============================================================================
print("1/5 Vectorizando descripciones geológicas...")
# Limpieza básica por si no se guardó en el parquet
df_ml['text_for_ml'] = df_ml['geology_of_the_overburden'].astype(str).str.lower().str.strip()

# Inicializamos el vectorizador
vectorizer = TfidfVectorizer(stop_words='english', min_df=4)

# fit_transform hace dos cosas: "aprende" el vocabulario y crea la matriz numérica
X_text = vectorizer.fit_transform(df_ml['text_for_ml'])

# ==============================================================================
# 3. ENTRENAMIENTO DEL MODELO (RANDOM FOREST)
# ==============================================================================
print("2/5 Entrenando clasificador Random Forest...")
# Variable objetivo (la que ya imputaste previamente en el notebook con K-Means)
y_train_res = df_ml['overburden_resistance_numeric'].astype(int)

# Entrenamos el modelo
rf_classifier = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
rf_classifier.fit(X_text, y_train_res)

# ==============================================================================
# 4. CÁLCULO DINÁMICO DE SUSCEPTIBILIDAD Y JENKS
# ==============================================================================
print("3/5 Calculando parámetros de normalización y umbrales...")
# Filtramos un subset limpio para recalcular el score correctamente
cols_score = ['depth_drift_clean', 'working_thickness_numeric', 'overburden_resistance_numeric']
df_score = modelo_granular.dropna(subset=cols_score).copy()

# Normalización Profundidad (inversa logarítmica)
log_depth = np.log1p(df_score['depth_drift_clean'])
log_min_depth, log_max_depth = log_depth.min(), log_depth.max()
norm_depth = (log_max_depth - log_depth) / (log_max_depth - log_min_depth)

# Normalización Espesor (lineal directa)
thick_min = df_score['working_thickness_numeric'].min()
thick_max = df_score['working_thickness_numeric'].max()
norm_thick = (df_score['working_thickness_numeric'] - thick_min) / (thick_max - thick_min)

# Multiplicador Resistencia
peso_resistencia = df_score['overburden_resistance_numeric'].map(CONFIG_RESISTENCIA)

# Cálculo Matemático Dinámico
df_score['Score_Dinamico'] = (norm_depth * CONFIG_PESOS['alpha']) + \
                             (norm_thick * CONFIG_PESOS['beta']) + \
                             (peso_resistencia * CONFIG_PESOS['gamma'])

# Jenks Natural Breaks sobre el score recién calculado
master_breaks = jenkspy.jenks_breaks(df_score['Score_Dinamico'].values, n_classes=3)

# ==============================================================================
# 5. MÓDULO DE MAGNITUD (CÁLCULO DINÁMICO DE LA MATRIZ CONDICIONAL)
# ==============================================================================
print("4/5 Calculando matriz de probabilidad condicional para Magnitud...")

# 5.1. Ahora filtramos y calculamos usando el dataset df_magnitud
columnas_modelo = ['Magnitud_Real_Peor_Caso', 'working_thickness_numeric', 'dip_clean']
df_modelo_final = df_magnitud.dropna(subset=columnas_modelo).copy()

# 5.2. Aplicamos pd.cut para crear las categorías discretas
df_modelo_final['Cat_Thickness'] = pd.cut(df_modelo_final['working_thickness_numeric'], 
                                          bins=CONFIG_BINS_MAGNITUD['thick_bins'], 
                                          labels=CONFIG_BINS_MAGNITUD['thick_labels'])

df_modelo_final['Cat_Dip'] = pd.cut(df_modelo_final['dip_clean'], 
                                    bins=CONFIG_BINS_MAGNITUD['dip_bins'], 
                                    labels=CONFIG_BINS_MAGNITUD['dip_labels'])

# 5.3. Construcción Matemática de la Matriz Condicional
matriz_conteos = df_modelo_final.groupby(['Cat_Thickness', 'Cat_Dip'], observed=False)['Magnitud_Real_Peor_Caso'].value_counts().unstack(fill_value=0)
matriz_prob_magnitud = matriz_conteos.div(matriz_conteos.sum(axis=1), axis=0) * 100
matriz_prob_magnitud = matriz_prob_magnitud.round(1).replace(0.0, np.nan).dropna(how='all')
matriz_prob_magnitud['Casos_Historicos'] = matriz_conteos.sum(axis=1)

# 5.4. Convertir el DataFrame resultante a un Diccionario para el motor (.pkl)
matriz_magnitud = {}
for idx, row in matriz_prob_magnitud.iterrows():
    cat_thick, cat_dip = idx
    matriz_magnitud[(cat_thick, cat_dip)] = {
        '1_Baja': row.get('1_Baja', 0) if pd.notna(row.get('1_Baja')) else 0,
        '2_Media': row.get('2_Media', 0) if pd.notna(row.get('2_Media')) else 0,
        '3_Alta': row.get('3_Alta', 0) if pd.notna(row.get('3_Alta')) else 0,
        'Casos_Historicos': int(row.get('Casos_Historicos', 0))
    }

# ==============================================================================
# 6. DICCIONARIO MAESTRO Y EXPORTACIÓN
# ==============================================================================
print("5/5 Empaquetando y exportando artefacto final...")
pipeline_completo = {
    'version': '2.0_SIRIMA',
    
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
    'umbrales_jenks': {'L1': master_breaks[1], 'L2': master_breaks[2]},

    # --- MÓDULO DE MAGNITUD ---
    'bins_magnitud': CONFIG_BINS_MAGNITUD,
    'matriz_magnitud': matriz_magnitud
}

ruta_archivo_final = os.path.join(ruta_empaquetado, 'pipeline_sirima_completo_renew.pkl')

with open(ruta_archivo_final, 'wb') as f:
    pickle.dump(pipeline_completo, f)

print(f"✅ ¡Éxito! Pipeline V3 exportado en: {ruta_archivo_final}")