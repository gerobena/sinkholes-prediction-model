import numpy as np
import pandas as pd
import pickle

# ==============================================================================
# 1. VISUAL PRINTING FUNCTION (FULL REPORT)
# ==============================================================================
def _print_hazard_panel(geology, resistance, depth, thickness, dip, susc_score, susc_level, expected_mag, max_prob, confidence, hazard_level):
    BOLD, CYAN, RESET = '\033[1m', '\033[96m', '\033[0m'
    
    # Susceptibility Traffic Light
    if susc_level == 'Low': c_susc = '\033[42m\033[97m'
    elif susc_level == 'Medium': c_susc = '\033[43m\033[30m'
    else: c_susc = '\033[41m\033[97m'
        
    # Magnitude Traffic Light and Translation
    mag_display = 'UNKNOWN'
    if '1_Baja' in expected_mag: 
        c_mag = '\033[42m\033[97m'
        mag_display = 'LOW'
    elif '2_Media' in expected_mag: 
        c_mag = '\033[43m\033[30m'
        mag_display = 'MEDIUM'
    elif '3_Alta' in expected_mag: 
        c_mag = '\033[41m\033[97m'
        mag_display = 'HIGH'
    else: 
        c_mag = '\033[47m\033[30m' # Dark grey for Missing Data
        mag_display = 'MISSING DATA'

    # Hazard Traffic Light (5 Levels)
    if 'Very Low' in hazard_level: c_pel = '\033[48;5;34m\033[97m' # Dark Green
    elif 'Low' in hazard_level: c_pel = '\033[42m\033[97m' # Green
    elif 'Moderate' in hazard_level: c_pel = '\033[43m\033[30m' # Yellow
    elif 'High' in hazard_level: c_pel = '\033[48;5;208m\033[97m' # Orange
    elif 'Critical' in hazard_level: c_pel = '\033[41m\033[97m' # Red
    else: c_pel = '\033[47m\033[30m'

    geo_str = geology if geology is not None else "N/A (Manual Entry)"

    print(f"\n{BOLD}===================================================={RESET}")
    print(f"{BOLD}      SIRIMA COMPREHENSIVE HAZARD REPORT            {RESET}")
    print(f"{BOLD}===================================================={RESET}")
    print(f"{BOLD}[INPUT DATA]{RESET}")
    print(f" Geology (Overburden Resistance):   {CYAN}{geo_str}{RESET} (Class {resistance})")
    print(f" Drift Depth:                       {CYAN}{depth:.1f} m{RESET}")
    print(f" Working Thickness:                 {CYAN}{thickness:.1f} m{RESET}")
    print(f" Dip:                               {CYAN}{dip:.1f}°{RESET}\n")
    
    print(f"{BOLD}[1. SUSCEPTIBILITY MODULE]{RESET}")
    print(f" Susceptibility Score:    {CYAN}{susc_score:.4f}{RESET}")
    print(f" Occurrence Prob.:      {BOLD}{c_susc} {susc_level.upper()} {RESET}\n")

    print(f"{BOLD}[2. MAGNITUDE MODULE (WORST-CASE SCENARIO)]{RESET}")
    if expected_mag == 'Sin Datos':
        print(f" Expected Magnitude:    {BOLD}{c_mag} MISSING HISTORICAL DATA {RESET}\n")
    else:
        print(f" Expected Magnitude:    {BOLD}{c_mag} {mag_display} {RESET} (Probability: {max_prob:.1f}% / Confidence: {confidence} cases)\n")

    print(f"{BOLD}[3. FINAL HAZARD ASSESSMENT]{RESET}")
    print(f" Hazard Index:          {BOLD}{c_pel} {hazard_level.upper()} {RESET}")
    print(f"{BOLD}===================================================={RESET}\n")


# ==============================================================================
# 2. SIRIMA ENGINE V3 (SUSCEPTIBILITY + MAGNITUDE + HAZARD)
# ==============================================================================
def evaluate_sirima_hazard(depth_m, thickness_m, dip_degrees, pipeline, geological_description=None, manual_resistance=None):
    try:
        # A. INITIAL VALIDATIONS
        if depth_m < 0 or thickness_m < 0 or dip_degrees < 0:
            raise ValueError("❌ PHYSICAL ERROR: Depth, thickness, and dip values cannot be negative.")
        if geological_description is not None and manual_resistance is not None:
            raise ValueError("❌ ERROR: Enter ONLY the geological description OR the manual resistance.")
        if geological_description is None and manual_resistance is None:
            raise ValueError("❌ ERROR: Provide a rock description or a resistance value (1, 2, 3).")

        # B. NLP MODULE: RESISTANCE CLASSIFICATION
        if manual_resistance is not None:
            resistance_class = int(manual_resistance)
        else:
            vec, classifier = pipeline['nlp_vectorizer'], pipeline['nlp_classifier']
            clean_text = str(geological_description).lower().strip()
            text_vector = vec.transform([clean_text])
            if text_vector.nnz == 0:
                raise ValueError("❌ NLP ERROR: The description does not contain recognized geological terms.")
            resistance_class = int(classifier.predict(text_vector)[0])

        # C. SUSCEPTIBILITY MODULE (SCORE AND LEVEL)
        params, weights = pipeline['parametros_normalizacion'], pipeline['pesos']        
        log_d = np.clip(np.log1p(depth_m), params['depth_log_min'], params['depth_log_max'])
        norm_depth = (params['depth_log_max'] - log_d) / (params['depth_log_max'] - params['depth_log_min'])
        t = np.clip(thickness_m, params['thick_min'], params['thick_max'])
        norm_thick = (t - params['thick_min']) / (params['thick_max'] - params['thick_min'])
        res_weight = pipeline['mapeo_res_a_peso'][resistance_class]
        
        susc_score = (norm_depth * weights['alpha']) + (norm_thick * weights['beta']) + (res_weight * weights['gamma'])
        
        thresholds = pipeline['umbrales_jenks']
        if susc_score <= thresholds['L1']: susc_level = 'Low'
        elif susc_score <= thresholds['L2']: susc_level = 'Medium'
        else: susc_level = 'High'

        # D. MAGNITUDE MODULE (CONDITIONAL PROBABILITY)
        def get_label(value, bins, labels):
            if value <= bins[0]: return labels[0]
            for i in range(len(bins)-1):
                if bins[i] < value <= bins[i+1]: return labels[i]
            return labels[-1]
            
        cat_thick = get_label(thickness_m, pipeline['bins_magnitud']['thick_bins'], pipeline['bins_magnitud']['thick_labels'])
        cat_dip = get_label(dip_degrees, pipeline['bins_magnitud']['dip_bins'], pipeline['bins_magnitud']['dip_labels'])
        #cat_res = f"Clase_{resistance_class}"

        scenario_key = (cat_thick, cat_dip)  #, cat_res)
        matrix = pipeline['matriz_magnitud']

        if scenario_key in matrix:
            probs = matrix[scenario_key]
            magnitudes = {
                '1_Baja': probs.get('1_Baja', 0) if not pd.isna(probs.get('1_Baja', 0)) else 0,
                '2_Media': probs.get('2_Media', 0) if not pd.isna(probs.get('2_Media', 0)) else 0,
                '3_Alta': probs.get('3_Alta', 0) if not pd.isna(probs.get('3_Alta', 0)) else 0
            }
            # Identify dominant magnitude
            expected_mag = max(magnitudes, key=magnitudes.get)
            max_prob = magnitudes[expected_mag] 
            confidence = int(probs.get('Casos_Historicos', 0))
        else:
            expected_mag = 'Sin Datos'
            max_prob = 0.0
            confidence = 0

        # E. COMPREHENSIVE HAZARD MATRIX (5-LEVEL LOGIC)
        susc_map = {'Low': 'Low', 'Medium': 'Medium', 'High': 'High'}
        expected_susc = susc_map[susc_level]

        hazard_level = 'Error'
        if expected_mag != 'Sin Datos':
            # 1. High Magnitude (M3)
            if expected_mag == '3_Alta':
                if expected_susc == 'High': hazard_level = 'Critical'
                elif expected_susc == 'Medium': hazard_level = 'High'
                elif expected_susc == 'Low': hazard_level = 'Moderate'
            
            # 2. Medium Magnitude (M2)
            elif expected_mag == '2_Media':
                if expected_susc == 'High': hazard_level = 'High'
                elif expected_susc == 'Medium': hazard_level = 'Moderate'
                elif expected_susc == 'Low': hazard_level = 'Low'
                
            # 3. Low Magnitude (M1)
            elif expected_mag == '1_Baja':
                if expected_susc == 'High': hazard_level = 'Moderate'
                elif expected_susc == 'Medium': hazard_level = 'Low'
                elif expected_susc == 'Low': hazard_level = 'Very Low'
        else:
            hazard_level = 'Undetermined (Missing Magnitude)'

        # F. PRINTING AND RETURN
        _print_hazard_panel(geological_description, resistance_class, depth_m, thickness_m, dip_degrees, susc_score, susc_level, expected_mag, max_prob, confidence, hazard_level)
        
        # The dictionary returned now also has standardized English keys
        return {
            'resistance_class': resistance_class,
            'susceptibility_score': susc_score,
            'susceptibility_level': susc_level,
            'expected_magnitude': expected_mag,
            'magnitude_probability': max_prob,
            'historical_cases': confidence,
            'hazard_level': hazard_level
        }

    except Exception as e:
        print(f"\n\033[91m\033[1m❌ ENGINE ERROR: {e}\033[0m\n")
        return None
