import numpy as np

# ==============================================================================
# 1. FUNCIÓN DE IMPRESIÓN VISUAL (INTERNA)
# ==============================================================================
def _imprimir_panel(geologia, resistencia, profundidad, espesor, score, nivel):
    BOLD, CYAN, RESET = '\033[1m', '\033[96m', '\033[0m'
    
    if nivel.upper() == 'LOW':
        COLOR_NIVEL = '\033[42m\033[97m'  # Verde
    elif nivel.upper() == 'MEDIUM':
        COLOR_NIVEL = '\033[43m\033[30m'  # Amarillo
    else: 
        COLOR_NIVEL = '\033[41m\033[97m'  # Rojo

    geo_str = geologia if geologia is not None else "N/A (Manual Resistance Entry)"

    print(f"\n{BOLD}--- PREDICTION SUMMARY ---{RESET}")
    print(f"{BOLD}Selected Geology:{RESET}                 {CYAN}{geo_str}{RESET}")
    print(f"{BOLD}Overburden Resistance:{RESET}            {CYAN}{resistencia}{RESET}")
    print(f"{BOLD}Depth of the Drift:{RESET}               {CYAN}{profundidad:.1f} m{RESET}")
    print(f"{BOLD}Coal Working Thickness:{RESET}           {CYAN}{espesor:.1f} m{RESET}")
    print(f"{BOLD}Calculated Susceptibility Score:{RESET}  {CYAN}{score:.4f}{RESET}")
    print(f"{BOLD}Susceptibility Level:{RESET}             {BOLD}{COLOR_NIVEL} {nivel.upper()} {RESET}\n")

# ==============================================================================
# 2. MOTOR OPERATIVO (CON MANEJO DE ERRORES INTERNO)
# ==============================================================================
def evaluar_susceptibilidad(profundidad_m, espesor_m, pipeline, descripcion_geologica=None, resistencia_manual=None):
    try:
        # A. VALIDACIÓN EXCLUYENTE
        if descripcion_geologica is not None and resistencia_manual is not None:
            raise ValueError("❌ ERROR: Ingresa SOLO la descripción geológica O la resistencia manual, no ambas.")
        if descripcion_geologica is None and resistencia_manual is None:
            raise ValueError("❌ ERROR: Faltan datos. Provee una descripción o un valor de resistencia (1, 2 o 3).")
        
        # B. PROCESAMIENTO NLP O MANUAL
        if resistencia_manual is not None:
            # BLOQUEO DE SEGURIDAD: Validamos estrictamente que sea 1, 2 o 3
            if str(resistencia_manual) not in ['1', '2', '3']:
                raise ValueError("❌ ERROR: Solo puede ingresar valores enteros 1, 2 y 3 como valores de resistencia o puede ingresar una descripción geológica en su lugar.")
            
            clase_resistencia = int(resistencia_manual)
            cluster_asignado = "Manual"
        else:
            vec, km, mapa = pipeline['nlp_vectorizer'], pipeline['nlp_kmeans'], pipeline['mapa_cluster_resistencia']
            texto_limpio = str(descripcion_geologica).lower().strip()
            vector_texto = vec.transform([texto_limpio])
            
            # Bloqueo (Solo cuando no existe ninguna palabra del catalogo, es decir, cuando todas las palabras son desconocidas)
            if vector_texto.nnz == 0:
                # nnz == 0 significa que el NLP no reconoció ni una sola palabra
                raise ValueError(
                    f"❌ ERROR NLP: La descripción ingresada no contiene términos geológicos reconocidos por el modelo historico ('{descripcion_geologica}'). Revise la ortografía en inglés."
                    )
                
            cluster_asignado = km.predict(vector_texto)[0]
            clase_resistencia = mapa[cluster_asignado] #mapa.get(cluster_asignado, 2)

        # C. NORMALIZACIÓN Y SCORE
        params, pesos = pipeline['parametros_normalizacion'], pipeline['pesos']        

        # 1. Normalizacion de profundidad reducido, segun el analisis anterior
            # 1.1. Tranformacion logaritmica para reducir el sesgo y cortar para los minimos y maximos del catalogo
        log_d = np.clip(np.log1p(profundidad_m), params['depth_log_min'], params['depth_log_max'])
            # 1.2. Normalizacion Min-Max Inversa sobre el algoritmo 
        norm_depth = (params['depth_log_max'] - log_d) / (params['depth_log_max'] - params['depth_log_min'])
        
        # 2. Normalizacion para espesor de capa
        # 2.1. Cortamos
        t = np.clip(espesor_m, params['thick_min'], params['thick_max'])
        # 2.2. Realizamos una normalización directa (Mayor espesor = Mayor Susceptibilidad)
        norm_thick = (t - params['thick_min']) / (params['thick_max'] - params['thick_min'])
        
        # 3. Normalizacion para la resistencia del material overburden
        # 3.1. Si se ingresa un valor manual diferente a 1, 2, 3. Asigna el valor de 0.35 (MEDIO) para la resistencia
        #peso_res = pipeline['mapeo_res_a_peso'].get(clase_resistencia, 0.35)
        # AHORA EL ACCESO ES ESTRICTO: Si no es 1, 2 o 3, el código fallará, sin valores por defecto inventados.
        peso_res = pipeline['mapeo_res_a_peso'][clase_resistencia]

        # 4. Calculo del Score Susceptibility (Suma ponderada)
        score = (norm_depth * pesos['alpha']) + (norm_thick * pesos['beta']) + (peso_res * pesos['gamma'])
        
        # D. CLASIFICACIÓN
        umbrales = pipeline['umbrales_jenks']
        if score <= umbrales['L1']: nivel = 'Low'
        elif score <= umbrales['L2']: nivel = 'Medium'
        else: nivel = 'High'
        
        # Si todo sale bien -> Imprime el panel visual automáticamente
        _imprimir_panel(descripcion_geologica, clase_resistencia, profundidad_m, espesor_m, score, nivel)
        return nivel, score, clase_resistencia, cluster_asignado

    except ValueError as e:
        print(f"\n\033[91m\033[1m{e}\033[0m\n")
        return "ERROR", 0.0, "N/A", "N/A"
    except KeyError:
        # Por si falla el mapeo interno del NLP de forma inesperada
        print("\n\033[91m\033[1m❌ ERROR INTERNO: El modelo NLP no pudo mapear el clúster a un nivel de resistencia válido.\033[0m\n")
        return "ERROR", 0.0, "N/A", "N/A"