import os
import pickle
import io
import re
from contextlib import redirect_stdout
import streamlit as st
from motor_sirima_completo import evaluate_sirima_hazard

# 1. Page Configuration
st.set_page_config(page_title="SIRIMA Hazard Model", page_icon="🚧", layout="centered")

st.title("SIRIMA System")
st.subheader("Comprehensive Hazard Assessment")

# 2. Cache to load the model only once and optimize performance
@st.cache_resource
def load_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'models', 'pipeline_sirima_completo.pkl')
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

pipeline = load_pipeline()

# 3. User Interface for input parameters
st.markdown("### Physical Parameters")
col1, col2 = st.columns(2)

with col1:
    depth_m = st.number_input("Drift Depth (m)", value=100.0, step=1.0)
    thickness_m = st.number_input("Working Thickness (m)", value=2.18, step=0.1)

with col2:
    dip_degrees = st.number_input("Dip (°)", value=2.0, step=1.0)

st.markdown("### Geological Parameters")
input_mode = st.radio("Select the resistance evaluation method:", 
                      ["Geological Description (NLP)", "Manual Resistance"])

geological_description = None
manual_resistance = None

if input_mode == "Geological Description (NLP)":
    geological_description = st.text_input("Enter the rock description (e.g. sands and shale):")
else:
    manual_resistance = st.selectbox("Select the Resistance Class:", [1, 2, 3])

st.markdown("---")

# 4. Execution Button
if st.button("Evaluate Hazard", type="primary", use_container_width=True):
    
    with st.spinner('Processing data in the SIRIMA engine...'):
        
        # Buffer to catch error logs from the engine
        terminal_output = io.StringIO()
        
        with redirect_stdout(terminal_output):
            results = evaluate_sirima_hazard(
                depth_m=depth_m,
                thickness_m=thickness_m,
                dip_degrees=dip_degrees,
                manual_resistance=manual_resistance,
                geological_description=geological_description,
                pipeline=pipeline
            )
        
        engine_message = terminal_output.getvalue()
        
        # 5. Display the results
        if results:
            st.success("✅ Evaluation completed successfully.")
            st.markdown("### Results Panel")
            
            # --- COLOR MAPPING LOGIC (Mirrors motor_sirima_completo.py) ---
            
            # A. Hazard Level Colors (5 Levels)
            hazard = results['hazard_level'].upper()
            if 'VERY LOW' in hazard: bg_h, text_h = "#1b5e20", "#ffffff"  # Dark Green
            elif 'LOW' in hazard: bg_h, text_h = "#2e7d32", "#ffffff"       # Green
            elif 'MODERATE' in hazard: bg_h, text_h = "#fbc02d", "#000000"  # Yellow
            elif 'HIGH' in hazard: bg_h, text_h = "#e65100", "#ffffff"      # Orange
            elif 'CRITICAL' in hazard: bg_h, text_h = "#c62828", "#ffffff"  # Red
            else: bg_h, text_h = "#757575", "#ffffff"                       # Grey
            
            # B. Susceptibility Level Colors
            susc = results['susceptibility_level'].upper()
            if 'LOW' in susc: bg_s, text_s = "#2e7d32", "#ffffff"
            elif 'MEDIUM' in susc: bg_s, text_s = "#fbc02d", "#000000"
            else: bg_s, text_s = "#c62828", "#ffffff"
            
            # C. Magnitude Level Colors & Translations
            mag_raw = results['expected_magnitude']
            if '1_Baja' in mag_raw: mag_display, bg_m, text_m = "LOW", "#2e7d32", "#ffffff"
            elif '2_Media' in mag_raw: mag_display, bg_m, text_m = "MEDIUM", "#fbc02d", "#000000"
            elif '3_Alta' in mag_raw: mag_display, bg_m, text_m = "HIGH", "#c62828", "#ffffff"
            else: mag_display, bg_m, text_m = "NO DATA", "#757575", "#ffffff"

            # --- DISPLAY VISUAL CARDS ---
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.markdown(f"""
                <div style="background-color: {bg_h}; padding: 15px; border-radius: 8px; text-align: center; min-height: 95px; box-shadow: 1px 1px 5px rgba(0,0,0,0.1);">
                    <div style="font-size: 11px; color: {text_h}; font-weight: 600; opacity: 0.85; letter-spacing: 0.5px;">HAZARD LEVEL</div>
                    <div style="font-size: 18px; font-weight: bold; color: {text_h}; margin-top: 8px;">{results['hazard_level'].upper()}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown(f"""
                <div style="background-color: {bg_s}; padding: 15px; border-radius: 8px; text-align: center; min-height: 95px; box-shadow: 1px 1px 5px rgba(0,0,0,0.1);">
                    <div style="font-size: 11px; color: {text_s}; font-weight: 600; opacity: 0.85; letter-spacing: 0.5px;">SUSCEPTIBILITY LEVEL</div>
                    <div style="font-size: 18px; font-weight: bold; color: {text_s}; margin-top: 8px;">{results['susceptibility_level'].upper()}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col3:
                st.markdown(f"""
                <div style="background-color: {bg_m}; padding: 15px; border-radius: 8px; text-align: center; min-height: 95px; box-shadow: 1px 1px 5px rgba(0,0,0,0.1);">
                    <div style="font-size: 11px; color: {text_m}; font-weight: 600; opacity: 0.85; letter-spacing: 0.5px;">MAGNITUDE LEVEL</div>
                    <div style="font-size: 18px; font-weight: bold; color: {text_m}; margin-top: 8px;">{mag_display}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # --- MODULE BREAKDOWN (Including Resistance Class) ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Module Breakdown")
            st.write(f"- **Resistance Class:** `{results['resistance_class']}`")
            st.write(f"- **Susceptibility Score:** `{results['susceptibility_score']:.4f}`")
            st.write(f"- **Scenario Probability:** `{results['magnitude_probability']:.1f}%`")
            st.write(f"- **Evaluated Historical Cases:** `{results['historical_cases']}`")
            
        else:
            # Clean ANSI color codes from the engine error message and display it
            clean_error = re.sub(r'\033\[[0-9;]*m', '', engine_message).strip()
            if clean_error:
                st.error(clean_error)
            else:
                st.error("❌ An unknown error occurred. No data was returned.")