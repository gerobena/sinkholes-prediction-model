import pickle
from motor_sirima_randomforest import evaluar_susceptibilidad_RF

# 1. Definir la ruta del modelo
ruta_modelo = 'models/pipeline_susceptibilidad_NLP_randomforest.pkl'

try:
    with open(ruta_modelo, 'rb') as f:
        pipeline_rf = pickle.load(f)
    print("✅ Pipeline cargado correctamente.\n")
except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo en {ruta_modelo}. Verifica la ruta.")
    exit()

# ==============================================================================
# PRUEBA 1: Flujo NLP (Simulando que el usuario de la web escribe texto)
# ==============================================================================
print("="*60)
print("👉 PRUEBA 1: Flujo de Texto Geológico (NLP)")
print("="*60)

res_1, metodo_1, score_1, nivel_1 = evaluar_susceptibilidad_RF(
    profundidad_m=20.0,
    espesor_m=2.4,
    descripcion_geologica="limestone",
    pipeline=pipeline_rf
)

# ==============================================================================
# PRUEBA 2: Flujo Manual (Simulando que el usuario selecciona la resistencia)
# ==============================================================================
print("="*60)
print("👉 PRUEBA 2: Flujo de Resistencia Manual")
print("="*60)

res_2, metodo_2, score_2, nivel_2 = evaluar_susceptibilidad_RF(
    profundidad_m=12.5,
    espesor_m=4.0,
    resistencia_manual=2,
    pipeline=pipeline_rf
)

# ==============================================================================
# PRUEBA 3: Manejo de Errores (Simulando datos negativos)
# ==============================================================================
print("="*60)
print("👉 PRUEBA 3: Validación de Errores (Profundidad negativa)")
print("="*60)

res_3, metodo_3, score_3, nivel_3 = evaluar_susceptibilidad_RF(
    profundidad_m=-5.0, # Dato inválido
    espesor_m=2.4,
    descripcion_geologica="clay",
    pipeline=pipeline_rf
)
