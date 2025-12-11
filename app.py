
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path so we can import jetx_project
sys.path.append(os.getcwd())

from jetx_project.config import WINDOWS, DB_LIMIT
from jetx_project.model_a import load_models, prepare_model_a_data
from jetx_project.features import extract_features
from jetx_project.model_b import load_memory, create_pattern_vector, predict_model_b
from jetx_project.model_lstm import load_lstm_models, create_sequences
from jetx_project.model_lightgbm import load_lightgbm_models
from jetx_project.model_mlp import load_mlp_models
from jetx_project.model_hmm import load_hmm_model, predict_hmm_state
from jetx_project.ensemble import load_meta_learner, prepare_meta_features, predict_meta
from jetx_project.model_transformer import load_transformer_models
import sqlite3

DB_PATH = "jetx.db"


def _ensure_table_exists(conn):
    """Create results table if it does not exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jetx_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            value REAL NOT NULL
        )
        """
    )


def save_to_db(value, db_path=DB_PATH):
    """Saves the new result to the database, raising on failure."""
    with sqlite3.connect(db_path, timeout=30) as conn:
        _ensure_table_exists(conn)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO jetx_results (value) VALUES (?)", (float(value),))
        conn.commit()
    return True

st.set_page_config(page_title="JetX Predictor", layout="wide")

st.title("JetX AI Prediction System (Ensemble Powered)")

# Load Models
# Load Models
@st.cache_resource
def load_all_models():
    models = {}
    
    # Fix: Use absolute path for model loading
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Model A
    try:
        ma_p15, ma_p3, ma_x = load_models(current_dir)
        models['model_a'] = {'p15': ma_p15, 'p3': ma_p3, 'x': ma_x} if ma_p15 else None
    except Exception as e:
        st.error(f"Model A failed to load: {e}")
        models['model_a'] = None

    # Model B
    try:
        mb_nbrs, mb_pca, mb_pats, mb_targs = load_memory(current_dir)
        models['model_b'] = {'nbrs': mb_nbrs, 'pca': mb_pca, 'targs': mb_targs} if mb_nbrs else None
    except Exception as e:
        st.error(f"Model B failed to load: {e}")
        models['model_b'] = None

    # Model C (LSTM)
    try:
        mc_p15, mc_p3, mc_scaler = load_lstm_models(current_dir)
        models['model_c'] = {'p15': mc_p15, 'p3': mc_p3, 'scaler': mc_scaler} if mc_p15 else None
    except Exception as e:
        st.error(f"Model C (LSTM) failed to load: {e}")
        models['model_c'] = None

    # Model D (LightGBM)
    try:
        md_p15, md_p3 = load_lightgbm_models(current_dir)
        models['model_d'] = {'p15': md_p15, 'p3': md_p3} if md_p15 else None
    except Exception as e:
        st.error(f"Model D (LightGBM) failed to load: {e}")
        models['model_d'] = None

    # Model E (MLP)
    try:
        me_p15, me_p3, me_cols = load_mlp_models(current_dir)
        models['model_e'] = {'p15': me_p15, 'p3': me_p3, 'cols': me_cols} if me_p15 else None
    except Exception as e:
        st.error(f"Model E (MLP) failed to load: {e}")
        models['model_e'] = None

    # HMM
    try:
        hmm_model, hmm_map, hmm_bins = load_hmm_model(current_dir)
        models['hmm'] = {'model': hmm_model, 'map': hmm_map, 'bins': hmm_bins} if hmm_model else None
    except Exception as e:
        st.error(f"HMM model failed to load: {e}")
        models['hmm'] = None

    # Meta Learner
    try:
        meta_model, meta_scaler = load_meta_learner(current_dir)
        models['meta'] = {'model': meta_model, 'scaler': meta_scaler} if meta_model else None
    except Exception as e:
        st.error(f"Meta Learner failed to load: {e}")
        models['meta'] = None
        
    # Transformer Model
    try:
        mt_model, mt_scaler = load_transformer_models(current_dir)
        models['transformer'] = {'model': mt_model, 'scaler': mt_scaler} if mt_model else None
    except Exception as e:
        st.error(f"Transformer Model failed to load: {e}")
        models['transformer'] = None
    
    return models

models = load_all_models()

if models is None:
    st.error("Please train models first using the Orchestrator.")
    st.stop()

model_name_map = {
    'model_a': 'Model A',
    'model_b': 'Model B',
    'model_c': 'Model C (LSTM)',
    'model_d': 'Model D (LightGBM)',
    'model_e': 'Model E (MLP)',
    'transformer': 'Transformer',
    'hmm': 'HMM',
    'meta': 'Meta-Learner'
}

loaded_predictors = [model_name_map[k] for k in ['model_a', 'model_b', 'model_c', 'model_d', 'model_e', 'transformer'] if models.get(k)]
loaded_context = [model_name_map[k] for k in ['hmm', 'meta'] if models.get(k)]

if not loaded_predictors and not loaded_context:
    st.error("Hiçbir model yüklenemedi. Lütfen Orchestrator ile modelleri eğitin.")
    st.stop()
else:
    st.success(f"Yüklenenler: {', '.join(loaded_predictors + loaded_context)}")

# Session State for History
if 'history' not in st.session_state:
    # Try to load from DB first
    if os.path.exists(DB_PATH):
        try:
            from jetx_project.data_loader import load_data, get_values_array
            from jetx_project.config import DB_LIMIT
            df = load_data(DB_PATH, limit=DB_LIMIT)
            vals = get_values_array(df)
            st.session_state.history = vals.tolist()
            st.success(f"Loaded last {len(vals)} records from jetx.db")
        except Exception as e:
            st.error(f"Could not load from DB: {e}")
            st.session_state.history = []
    else:
        st.session_state.history = [] 
        
# Enforce RAM Limit on Startup
if len(st.session_state.history) > DB_LIMIT:
    st.session_state.history = st.session_state.history[-DB_LIMIT:] 

# Sidebar Navigation
page = st.sidebar.radio("Menü", ["🚀 Canlı Tahmin", "📥 Toplu Veri Girişi"])

if page == "🚀 Canlı Tahmin":
    # ---------------------------------------------------------
    # EXISTING LIVE PREDICTOR PAGE
    # ---------------------------------------------------------
    
    # Input
    new_val = st.number_input("Enter Last Result (X):", min_value=1.00, max_value=100000.0, step=0.01, format="%.2f")

    if st.button("Add Result & Predict"):
        # 1. Save to DB (Persistence) - CRITICAL FIX: Only update RAM if DB save succeeds
        try:
            save_to_db(new_val, DB_PATH)
        except Exception as e:
            st.error(f"CRITICAL: Failed to save to DB. RAM not updated to ensure consistency. Error: {e}")
            st.stop()
        
        # 2. Update Session State (only if DB write succeeded)
        # Fix: Moved inside the success block to ensure consistency
        st.session_state.history.append(new_val)
        
        # CRITICAL FIX: Memory Leak & Sync
        # Enforce strict limit on RAM history to match DB_LIMIT
        if len(st.session_state.history) > DB_LIMIT:
            st.session_state.history = st.session_state.history[-DB_LIMIT:]
        
        history_arr = np.array(st.session_state.history)
        current_idx = len(history_arr) - 1
        
        # Check for sufficient data
        if len(history_arr) < 500:
            st.warning(f"Not enough data for reliable predictions. Collecting data... ({len(history_arr)}/500)")
            st.stop()
        
        # Initialize probabilities
        # Fix: Initialize with None to avoid biased average with 0.0s
        probs = {
            'A': None, 'B': None, 'C': None, 'D': None, 'E': None, 'T': None
        }
        
        # --- HMM State (causal, early) ---
        try:
            hmm_window = 500
            hmm_input = history_arr[-hmm_window:] if len(history_arr) > hmm_window else history_arr
            from jetx_project.model_hmm import predict_categorical_hmm_states_causal
            if models.get('hmm'):
                hmm_model = models['hmm']['model']
                hmm_map = models['hmm']['map']
                hmm_bins = models['hmm']['bins']
                hmm_states = predict_categorical_hmm_states_causal(hmm_model, hmm_input, hmm_map, bins=hmm_bins, window_size=200)
                current_state = hmm_states[-1] if len(hmm_states) > 0 else None
            else:
                current_state = None
        except Exception as e:
            st.error(f"HMM Error: {e}")
            st.warning("⚠️ Piyasa rejimi tespit edilemedi, güvenlik modu devrede.")
            current_state = None
        if current_state is None:
            current_state = 1  # neutral fallback

        # --- Feature Extraction ---
        if len(history_arr) >= 500:
            try:
                feats = extract_features(history_arr, current_idx)
                feats_df = pd.DataFrame([feats])
                feats_df['hmm_state'] = current_state
                
                # Model A
                if models.get('model_a'):
                    probs['A'] = models['model_a']['p15'].predict_proba(feats_df)[0][1]
                
                # Model D
                if models.get('model_d'):
                    probs['D'] = models['model_d']['p15'].predict_proba(feats_df)[0][1]
                
                # Model E
                if models.get('model_e'):
                    me_cols = models['model_e']['cols']
                    feats_mlp = feats_df[me_cols]
                    probs['E'] = models['model_e']['p15'].predict_proba(feats_mlp)[0][1]
                
            except Exception as e:
                st.error(f"Feature Extraction Error: {e}")

        # --- Pattern Recognition (Model B) ---
        if len(history_arr) >= 300 and models.get('model_b'):
            try:
                mb_nbrs = models['model_b']['nbrs']
                mb_pca = models['model_b']['pca']
                mb_targs = models['model_b']['targs']
                
                pat = create_pattern_vector(history_arr, current_idx)
                if pat is not None:
                    pat_reshaped = pat.reshape(1, -1)
                    probs['B'], _, _ = predict_model_b(mb_nbrs, mb_pca, mb_targs, pat_reshaped)
            except Exception as e:
                st.error(f"Model B Error: {e}")
                
        # --- LSTM (Model C) ---
        if len(history_arr) >= 200 and models.get('model_c'):
            try:
                mc_p15 = models['model_c']['p15']
                mc_scaler = models['model_c']['scaler']
                
                seq_len = 200
                last_seq = history_arr[-seq_len:]
                last_seq_log = np.log1p(last_seq)
                last_seq_scaled = mc_scaler.transform(last_seq_log.reshape(-1, 1))
                last_seq_scaled = np.clip(last_seq_scaled, 0, 1)
                X_lstm = last_seq_scaled.reshape(1, seq_len, 1)
                probs['C'] = mc_p15.predict(X_lstm)[0][0]
            except Exception as e:
                st.error(f"Model C Error: {e}")

        # --- Transformer (Model T) ---
        if len(history_arr) >= 200 and models.get('transformer'):
            try:
                mt_model = models['transformer']['model']
                mt_scaler = models['transformer']['scaler']

                seq_len = 200
                last_seq = history_arr[-seq_len:]
                last_seq_log = np.log1p(last_seq)
                last_seq_scaled = mt_scaler.transform(last_seq_log.reshape(-1, 1))
                last_seq_scaled = np.clip(last_seq_scaled, 0, 1)
                X_transformer = last_seq_scaled.reshape(1, seq_len, 1)

                preds_t = mt_model.predict(X_transformer)
                if isinstance(preds_t, (list, tuple)) and len(preds_t) > 0:
                    transformer_prob = float(np.ravel(preds_t[0])[0])
                else:
                    transformer_prob = float(np.ravel(preds_t)[0])
                probs['T'] = transformer_prob
            except Exception as e:
                st.error(f"Transformer Model Error: {e}")
                
        # --- ENSEMBLE PREDICTION ---
        # Prepare Meta Features
        # Fix: Wrap in Try/Except for Robustness
        try:
            real_history = np.array(st.session_state.history[-250:]) if st.session_state.history else np.array([])
            # Retrieve HMM component count if available
            n_hmm_components = 3
            if models.get('hmm') and hasattr(models['hmm']['model'], 'n_components'):
                n_hmm_components = models['hmm']['model'].n_components

            meta_X = prepare_meta_features(
                preds_a=np.array([probs['A'] if probs['A'] is not None else 0.5]),
                preds_b=np.array([probs['B'] if probs['B'] is not None else 0.5]),
                preds_c=np.array([probs['C'] if probs['C'] is not None else 0.5]),
                preds_d=np.array([probs['D'] if probs['D'] is not None else 0.5]),
                preds_e=np.array([probs['E'] if probs['E'] is not None else 0.5]),
                hmm_states=np.array([current_state]),
                values=real_history,
                preds_transformer=np.array([probs['T'] if probs['T'] is not None else 0.5]) if probs['T'] is not None else None,
                n_hmm_components=n_hmm_components
            )
            
            if models.get('meta'):
                final_prob = predict_meta(models['meta']['model'], models['meta']['scaler'], meta_X)[0]
            else:
                # Soft Voting Fallback
                available_probs = [p for p in probs.values() if p is not None]
                if available_probs:
                    final_prob = np.mean(available_probs)
                else:
                    st.error("Hiçbir model tahmin üretemedi!")
                    final_prob = 0.0
                    
        except Exception as e:
            st.error(f"Ensemble/Meta-Learner Error: {e}")
            st.warning("⚠️ Falling back to simple average due to ensemble error.")
            available_probs = [p for p in probs.values() if p is not None]
            final_prob = np.mean(available_probs) if available_probs else 0.0
        
        # --- Display Results ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Model Predictions (1.5x)")
            if probs['A'] is not None: st.write(f"CatBoost (A): {probs['A']:.2%}")
            else: st.write("CatBoost (A): N/A")
            
            if probs['B'] is not None: st.write(f"Memory (B): {probs['B']:.2%}")
            else: st.write("Memory (B): N/A")
            
            if probs['C'] is not None: st.write(f"LSTM (C): {probs['C']:.2%}")
            else: st.write("LSTM (C): N/A")
            
            if probs['D'] is not None: st.write(f"LightGBM (D): {probs['D']:.2%}")
            else: st.write("LightGBM (D): N/A")
            
            if probs['E'] is not None: st.write(f"MLP (E): {probs['E']:.2%}")
            else: st.write("MLP (E): N/A")
            
            if probs['T'] is not None: st.write(f"Transformer (T): {probs['T']:.2%}")
            else: st.write("Transformer (T): N/A")
            
        with col2:
            st.subheader("Market Context")
            state_names = {0: "Cold (Düşük)", 1: "Normal", 2: "Hot (Yüksek)"}
            st.metric("HMM Regime", state_names.get(current_state, "Unknown"))
            
        with col3:
            st.subheader("ENSEMBLE DECISION")
            st.metric("Final Probability (>1.5x)", f"{final_prob:.2%}", 
                     delta="High Confidence" if final_prob > 0.70 else "Low Confidence")
            
            if final_prob > 0.70:
                st.success("🚀 SIGNAL: BET (1.50x)")
            else:
                st.error("🛑 SIGNAL: WAIT")

    # Show History
    st.subheader("History")
    st.write(st.session_state.history[-20:])

elif page == "📥 Toplu Veri Girişi":
    # ---------------------------------------------------------
    # NEW BULK IMPORT PAGE
    # ---------------------------------------------------------
    st.header("📥 Toplu Veri Girişi (Bulk Import)")
    
    st.info("""
    **ℹ️ NASIL ÇALIŞIR?**
    
    Bu panel, eksik kalan geçmiş oyun verilerini sisteme hızlıca eklemenizi sağlar.
    
    **⚠️ ÖNEMLİ (GİRİŞ SIRASI):**
    1.  Listeyi kopyaladığınız gibi yapıştırın.
    2.  **EN ÜSTTEKİ** veri **EN GÜNCEL (YENİ)** veri olmalıdır.
    3.  **EN ALTTAKİ** veri **EN ESKİ** veri olmalıdır.
    4.  Sistem bu listeyi otomatik olarak ters çevirip doğru sırayla (Eskiden Yeniye) veritabanına ekleyecektir.
    
    _Verileri virgül, boşluk veya yeni satır ile ayırabilirsiniz._ (x harfi otomatik temizlenir)
    """)
    
    raw_input = st.text_area("Verileri Yapıştırın (En Üst = En Yeni)", height=200)
    
    if st.button("Verileri İçe Aktar"):
        import re
        
        # 1. Parse Input
        # Replace commas/newlines/x with spaces, then split
        clean_text = raw_input.replace(',', ' ').replace('\n', ' ').replace('x', '').replace('X', '')
        tokens = clean_text.split()
        
        valid_values = []
        try:
            for t in tokens:
                val = float(t.strip())
                if val >= 1.0:
                    valid_values.append(val)
        except Exception as e:
            st.error(f"Veri ayrıştırma hatası: {e}")
            st.stop()
            
        if not valid_values:
            st.warning("Lütfen geçerli sayısal değerler girin.")
            st.stop()
            
        # REVERSE LIST: User input is Newest -> Oldest. We need Chronological (Oldest -> Newest) for DB.
        valid_values.reverse()
        
        # 2. Save to DB & Session State loop
        success_count = 0
        fail_count = 0
        
        progress_bar = st.progress(0)
        
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            _ensure_table_exists(conn)
            cursor = conn.cursor()
            
            # Use a list to collect values that are successfully staged for DB
            staged_values = []
            
            for i, val in enumerate(valid_values):
                try:
                    cursor.execute("INSERT INTO jetx_results (value) VALUES (?)", (float(val),))
                    staged_values.append(val)
                    success_count += 1
                except Exception as e:
                    fail_count += 1
                    print(f"DB Insert Error for {val}: {e}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(valid_values))
            
            # Critical Fix: Commit first, then update RAM
            try:
                conn.commit()
                # Only if commit succeeds, we update the RAM history
                st.session_state.history.extend(staged_values)
            except Exception as e:
                st.error(f"FATAL: Database Commit Failed! RAM was NOT updated. Error: {e}")
                # Reset success count because nothing was actually saved
                success_count = 0
                fail_count = len(valid_values)
            
        # 3. Enforce Limits
        if len(st.session_state.history) > DB_LIMIT:
            st.session_state.history = st.session_state.history[-DB_LIMIT:]
            
        if success_count > 0:
            st.success(f"✅ {success_count} adet veri başarıyla eklendi! (Hatalı: {fail_count})")
            st.write("Eklenen son 5 değer:", valid_values[-5:])
            st.info("Modeller artık bu yeni verileri geçmişi olarak görecektir.")
        else:
            st.error("Hiçbir veri eklenemedi.")
