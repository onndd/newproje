
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
from jetx_project.model_lstm import load_lstm_models, create_sequences
from jetx_project.model_lightgbm import load_lightgbm_models
from jetx_project.model_mlp import load_mlp_models
from jetx_project.model_hmm import load_hmm_model, predict_hmm_state
from jetx_project.ensemble import load_meta_learner, prepare_meta_features, predict_meta

st.set_page_config(page_title="JetX Predictor", layout="wide")

st.title("JetX AI Prediction System (Ensemble Powered)")

# Load Models
@st.cache_resource
def load_all_models():
    try:
        # Load all individual models
        ma_p15, ma_p3, ma_x = load_models('.')
        mb_nbrs, mb_pca, mb_pats, mb_targs = load_memory('.')
        mc_p15, mc_p3, mc_scaler = load_lstm_models('.')
        md_p15, md_p3 = load_lightgbm_models('.')
        me_p15, me_p3, me_cols = load_mlp_models('.')
        # Updated to load bins for CategoricalHMM
        hmm_model, hmm_map, hmm_bins = load_hmm_model('.')
        meta_model, meta_scaler = load_meta_learner('.')
        
        return (ma_p15, ma_p3, ma_x, 
                mb_nbrs, mb_pca, mb_targs, 
                mc_p15, mc_p3, mc_scaler,
                md_p15, md_p3,
                me_p15, me_p3, me_cols,
                hmm_model, hmm_map, hmm_bins,
                meta_model, meta_scaler)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

models = load_all_models()

if models:
    (ma_p15, ma_p3, ma_x, 
     mb_nbrs, mb_pca, mb_targs, 
     mc_p15, mc_p3, mc_scaler,
     md_p15, md_p3,
     me_p15, me_p3, me_cols,
     hmm_model, hmm_map, hmm_bins,
     meta_model, meta_scaler) = models
    st.success("All Models & Ensemble Loaded Successfully!")
else:
    st.warning("Please train models first using the Orchestrator.")
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
    
    # Initialize probabilities
    probs = {
        'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0
    }
    
    # --- Feature Extraction ---
    if len(history_arr) >= 500:
        try:
            feats = extract_features(history_arr, current_idx)
            feats_df = pd.DataFrame([feats])
            
            # Model A
            probs['A'] = ma_p15.predict_proba(feats_df)[0][1]
            
            # Model D
            probs['D'] = md_p15.predict_proba(feats_df)[0][1]
            
            # Model E
            feats_mlp = feats_df[me_cols]
            probs['E'] = me_p15.predict_proba(feats_mlp)[0][1]
            
        except Exception as e:
            st.error(f"Feature Extraction Error: {e}")

    # --- Pattern Recognition (Model B) ---
    if len(history_arr) >= 300:
        try:
            pat = create_pattern_vector(history_arr, current_idx)
            if pat is not None:
                pat_reshaped = pat.reshape(1, -1)
                probs['B'], _, _ = predict_model_b(mb_nbrs, mb_pca, mb_targs, pat_reshaped)
        except Exception as e:
            st.error(f"Model B Error: {e}")
            
    # --- LSTM (Model C) ---
    if len(history_arr) >= 200:
        try:
            seq_len = 200
            # Get last seq_len values
            last_seq = history_arr[-seq_len:]
            # Scale and Clip
            last_seq_scaled = mc_scaler.transform(last_seq.reshape(-1, 1))
            last_seq_scaled = np.clip(last_seq_scaled, 0, 1) # Ensure within bounds
            X_lstm = last_seq_scaled.reshape(1, seq_len, 1)
            probs['C'] = mc_p15.predict(X_lstm)[0][0]
        except Exception as e:
            st.error(f"Model C Error: {e}")
            
    # --- HMM State ---
    try:
        # Predict state using CategoricalHMM
        # Ideally we just need the last state, but HMM is sequential.
        # For speed, we can take a recent window if history is huge.
        hmm_window = 500
        if len(history_arr) > hmm_window:
            hmm_input = history_arr[-hmm_window:]
        else:
            hmm_input = history_arr
            
        # Use CategoricalHMM prediction
        from jetx_project.model_hmm import predict_categorical_hmm_states
        hmm_states = predict_categorical_hmm_states(hmm_model, hmm_input, hmm_map, bins=hmm_bins)
        current_state = hmm_states[-1]
    except Exception as e:
        st.error(f"HMM Error: {e}")
        current_state = 1 # Default to Normal
        
    # --- ENSEMBLE PREDICTION ---
    # Prepare meta features
    # Order: A, B, C, D, E, HMM_State
    # Note: prepare_meta_features expects arrays, so we wrap in list
    
    meta_X = prepare_meta_features(
        np.array([probs['A']]),
        np.array([probs['B']]),
        np.array([probs['C']]),
        np.array([probs['D']]),
        np.array([probs['E']]),
        np.array([current_state]),
        values=history_arr # Pass raw values for 1.00x frequency feature
    )
    
    final_prob = predict_meta(meta_model, meta_scaler, meta_X)[0]
    
    # --- Display Results ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Individual Models")
        st.write(f"CatBoost (A): {probs['A']:.2%}")
        st.write(f"k-NN (B): {probs['B']:.2%}")
        st.write(f"LSTM (C): {probs['C']:.2%}")
        st.write(f"LightGBM (D): {probs['D']:.2%}")
        st.write(f"MLP (E): {probs['E']:.2%}")
        
    with col2:
        st.subheader("Market Context")
        state_names = {0: "Cold (Düşük)", 1: "Normal", 2: "Hot (Yüksek)"}
        st.metric("HMM Regime", state_names.get(current_state, "Unknown"))
        
    with col3:
        st.subheader("ENSEMBLE DECISION")
        st.metric("Final Probability (>1.5x)", f"{final_prob:.2%}", 
                 delta="High Confidence" if final_prob > 0.65 else "Low Confidence")
        
        if final_prob > 0.65:
            st.success("🚀 SIGNAL: BET (1.50x)")
        else:
            st.error("🛑 SIGNAL: WAIT")

# Show History
st.subheader("History")
st.write(st.session_state.history[-20:])
