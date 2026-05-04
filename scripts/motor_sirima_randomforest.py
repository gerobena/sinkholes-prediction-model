import numpy as np

# ==============================================================================
# 1. FUNCIÓN DE IMPRESIÓN VISUAL (INTERNA) - [SE MANTIENE IGUAL]
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
# 2. MOTOR OPERATIVO (BLINDADO A PRUEBA DE FALLOS)
# ==============================================================================
def evaluar_susceptibilidad_RF(profundidad_m, espesor_m, pipeline, descripcion_geologica=None, resistencia_manual=None):
    try:
        # A. VALIDACIONES INICIALES
        if profundidad_m < 0 or espesor_m < 0:
            raise ValueError("❌ ERROR FÍSICO: La profundidad y el espesor no pueden ser valores negativos.")
            
        if descripcion_geologica is not None and resistencia_manual is not None:
            raise ValueError("❌ ERROR: Ingresa SOLO la descripción geológica O la resistencia manual, no ambas.")
            
        if descripcion_geologica is None and resistencia_manual is None:
            raise ValueError("❌ ERROR: Faltan datos. Provee una descripción o un valor de resistencia (1, 2 o 3).")
        
        # B. PROCESAMIENTO NLP O MANUAL
        if resistencia_manual is not None:
            if str(resistencia_manual) not in ['1', '2', '3']:
                raise ValueError("❌ ERROR: Solo puede ingresar valores enteros 1, 2 y 3 como valores de resistencia o puede ingresar una descripción geológica en su lugar.")
            
            clase_resistencia = int(resistencia_manual)
            metodo_asignacion = "Manual"
        else:
            vec = pipeline['nlp_vectorizer']
            clasificador = pipeline['nlp_classifier']
            
            texto_limpio = str(descripcion_geologica).lower().strip()
            vector_texto = vec.transform([texto_limpio])
            
            if vector_texto.nnz == 0:
                raise ValueError(
                    f"❌ ERROR NLP: La descripción ingresada no contiene términos geológicos reconocidos ('{descripcion_geologica}'). Revise la ortografía en inglés."
                    )
                
            clase_resistencia = int(clasificador.predict(vector_texto)[0])
            metodo_asignacion = "NLP_RandomForest"

        # C. NORMALIZACIÓN Y SCORE
        params, pesos = pipeline['parametros_normalizacion'], pipeline['pesos']        

        # 1. Usamos clip() para definir los limites del modelo. Es decir si se ingresan parametros mayores a los establecidos por le modelo, tomará el mayor valor para realizar el calculo
        log_d = np.clip(np.log1p(profundidad_m), params['depth_log_min'], params['depth_log_max'])

        # 2. Normalizacion de Depth Relative to the Surface 
        norm_depth = (params['depth_log_max'] - log_d) / (params['depth_log_max'] - params['depth_log_min'])
        
        # 3. Normalizacion de Working Thickness
        t = np.clip(espesor_m, params['thick_min'], params['thick_max'])
        norm_thick = (t - params['thick_min']) / (params['thick_max'] - params['thick_min'])
        
        # Acceso estricto (Es seguro porque validamos que siempre sea 1, 2 o 3)
        peso_res = pipeline['mapeo_res_a_peso'][clase_resistencia]

        # Calculo del Susceptibility Score (Suma ponderada de varibales nomralizadas y pesos generales)
        score = (norm_depth * pesos['alpha']) + (norm_thick * pesos['beta']) + (peso_res * pesos['gamma'])
        
        # D. CLASIFICACIÓN
        umbrales = pipeline['umbrales_jenks']
        if score <= umbrales['L1']: nivel = 'Low'
        elif score <= umbrales['L2']: nivel = 'Medium'
        else: nivel = 'High'
        
        _imprimir_panel(descripcion_geologica, clase_resistencia, profundidad_m, espesor_m, score, nivel)
        
        # RETORNO CONSISTENTE ÉXITO (4 variables)
        return clase_resistencia, metodo_asignacion, score, nivel

    except ValueError as e:
        print(f"\n\033[91m\033[1m{e}\033[0m\n")
        # RETORNO CONSISTENTE ERROR (4 variables)
        return "ERROR", "N/A", None, "N/A"
        
    except KeyError as e:
        print(f"\n\033[91m\033[1m❌ ERROR INTERNO: Llave faltante en el pipeline ({e}). Verifica el archivo .pkl.\033[0m\n")
        # RETORNO CONSISTENTE ERROR (4 variables)
        return "ERROR", "N/A", None, "N/A"