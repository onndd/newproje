
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path so we can import jetx_project
sys.path.append(os.getcwd())

from jetx_project.config import WINDOWS
from jetx_project.model_a import load_models, prepare_model_a_data
from jetx_project.features import extract_features
from jetx_project.model_b import load_memory, create_pattern_vector, predict_model_b

st.set_page_config(page_title="JetX Predictor", layout="wide")

st.title("JetX AI Prediction System")

# Load Models
@st.cache_resource
def load_all_models():
    try:
        ma_p15, ma_p3, ma_x = load_models('.')
        mb_nbrs, mb_pats, mb_targs = load_memory('.')
        return ma_p15, ma_p3, ma_x, mb_nbrs, mb_pats, mb_targs
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

models = load_all_models()

if models:
    ma_p15, ma_p3, ma_x, mb_nbrs, mb_pats, mb_targs = models
    st.success("Models loaded successfully!")
else:
    st.warning("Please train models first using the Orchestrator Notebook.")
    st.stop()

# Session State for History
if 'history' not in st.session_state:
    # Try to load from DB first
    if os.path.exists('jetx.db'):
        try:
            from jetx_project.data_loader import load_data, get_values_array
            df = load_data('jetx.db')
            vals = get_values_array(df)
            st.session_state.history = vals.tolist()
            st.success(f"Loaded {len(vals)} records from jetx.db")
        except Exception as e:
            st.error(f"Could not load from DB: {e}")
            st.session_state.history = []
    else:
        st.session_state.history = [] 

# Input
new_val = st.number_input("Enter Last Result (X):", min_value=1.00, step=0.01, format="%.2f")

if st.button("Add Result & Predict"):
    st.session_state.history.append(new_val)
    
    history_arr = np.array(st.session_state.history)
    current_idx = len(history_arr) - 1
    
    # Initialize predictions to 0
    pred_a_p15, pred_a_p3, pred_a_x = 0.0, 0.0, 0.0
    pred_b_p15, pred_b_p3, pred_b_x = 0.0, 0.0, 0.0
    
    model_a_active = False
    model_b_active = False

    # --- Model A Prediction (Needs 500 data points) ---
    if len(history_arr) >= 500:
        try:
            feats = extract_features(history_arr, current_idx)
            feats_df = pd.DataFrame([feats])
            
            pred_a_p15 = ma_p15.predict_proba(feats_df)[0][1]
            pred_a_p3 = ma_p3.predict_proba(feats_df)[0][1]
            pred_a_x = ma_x.predict(feats_df)[0]
            model_a_active = True
        except Exception as e:
            st.error(f"Model A Error: {e}")
    
    # --- Model B Prediction (Needs 300 data points) ---
    if len(history_arr) >= 300:
        try:
            pat = create_pattern_vector(history_arr, current_idx)
            # Reshape for single prediction
            pat_reshaped = pat.reshape(1, -1)
            pred_b_p15, pred_b_p3, pred_b_x = predict_model_b(mb_nbrs, mb_targs, pat_reshaped)
            model_b_active = True
        except Exception as e:
            st.error(f"Model B Error: {e}")

    # --- Display Results ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Model A (Feature)")
        if model_a_active:
            st.metric("P(>1.5)", f"{pred_a_p15:.2%}", delta_color="normal" if pred_a_p15 < 0.8 else "inverse")
            st.metric("P(>3.0)", f"{pred_a_p3:.2%}")
            st.metric("Pred X", f"{pred_a_x:.2f}")
        else:
            st.warning("Need 500+ data points")
            
    with col2:
        st.subheader("Model B (Pattern)")
        if model_b_active:
            st.metric("P(>1.5)", f"{pred_b_p15:.2%}")
            st.metric("P(>3.0)", f"{pred_b_p3:.2%}")
            st.metric("Pred X", f"{pred_b_x:.2f}")
        else:
            st.warning("Need 300+ data points")

    with col3:
        st.subheader("Ensemble / Decision")
        
        # Calculate Average based on active models
        if model_a_active and model_b_active:
            avg_p15 = (pred_a_p15 + pred_b_p15) / 2
            avg_p3 = (pred_a_p3 + pred_b_p3) / 2
        elif model_a_active:
            avg_p15 = pred_a_p15
            avg_p3 = pred_a_p3
        elif model_b_active:
            avg_p15 = pred_b_p15
            avg_p3 = pred_b_p3
        else:
            avg_p15 = 0.0
            avg_p3 = 0.0
            
        if model_a_active or model_b_active:
            st.metric("Avg P(>1.5)", f"{avg_p15:.2%}")
            
            if avg_p15 > 0.8:
                st.success("SIGNAL: SAFE BET (1.5x)")
            elif avg_p3 > 0.6: 
                st.warning("SIGNAL: RISK BET (3x)")
            else:
                st.info("NO SIGNAL")
        else:
            st.write("Waiting for more data...")

# Show History
st.subheader("History")
st.write(st.session_state.history[-20:])
