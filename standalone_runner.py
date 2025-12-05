import os
import sys
import numpy as np
import pandas as pd
import sqlite3
import joblib

# Add current directory to path
sys.path.append(os.getcwd())

# Import Project Modules
from jetx_project.config import DB_PATH, HMM_BIN_EDGES
from jetx_project.data_loader import load_data, get_values_array
from jetx_project.model_a import prepare_model_a_data, train_model_a, save_models
from jetx_project.model_b import build_memory, train_model_b, save_memory
from jetx_project.model_lstm import train_model_lstm, save_lstm_models
from jetx_project.model_lightgbm import train_model_lightgbm, save_lightgbm_models
from jetx_project.model_mlp import train_model_mlp, save_mlp_models
from jetx_project.model_hmm import train_categorical_hmm, save_hmm_model, predict_categorical_hmm_states
from jetx_project.model_transformer import train_model_transformer, save_transformer_models
from jetx_project.ensemble import prepare_meta_features, train_meta_learner, save_meta_learner
from jetx_project.simulation import run_simulation

def main():
    print("🚀 JetX Standalone Runner Initiated...")
    
    # 1. Load Data
    print("\n[1/9] Loading Data...")
    if not os.path.exists(DB_PATH):
        print(f"Error: Database {DB_PATH} not found. Please provide data.")
        return
        
    df = load_data(DB_PATH, limit=10000) # Load ample data for training
    values = get_values_array(df)
    print(f"Loaded {len(values)} records.")
    
    if len(values) < 1000:
        print("Error: Not enough data to train (need > 1000).")
        return

    # 2. Train HMM (First, as others need states)
    print("\n[2/9] Training HMM (Categorical)...")
    # Fix: Use train_categorical_hmm to match the 3 return values expected (model, map, bins)
    hmm_model, hmm_map, hmm_bins = train_categorical_hmm(values)
    save_hmm_model(hmm_model, hmm_map, hmm_bins, output_dir='models_standalone')
    
    # Predict states for all data
    hmm_states = predict_categorical_hmm_states(hmm_model, values, hmm_map, bins=hmm_bins)
    
    # 3. Train Model A (CatBoost)
    print("\n[3/9] Training Model A (CatBoost)...")
    X_a, y_p15_a, y_p3_a, y_x_a = prepare_model_a_data(values, hmm_states)
    ma_p15, ma_p3, ma_x = train_model_a(X_a, y_p15_a, y_p3_a, y_x_a)
    save_models(ma_p15, ma_p3, ma_x, output_dir='models_standalone')
    
    # 4. Train Model B (Memory)
    print("\n[4/9] Training Model B (Memory)...")
    patterns, targets = build_memory(values)
    nbrs, pca = train_model_b(patterns)
    save_memory(nbrs, pca, patterns, targets, output_dir='models_standalone')
    
    # 5. Train Model C (LSTM)
    print("\n[5/9] Training Model C (LSTM)...")
    mc_p15, mc_p3, mc_scaler = train_model_lstm(values)
    save_lstm_models(mc_p15, mc_p3, mc_scaler, output_dir='models_standalone')
    
    # 6. Train Model D (LightGBM)
    print("\n[6/9] Training Model D (LightGBM)...")
    # Reuse X_a as it has the same features
    md_p15, md_p3 = train_model_lightgbm(X_a, y_p15_a, y_p3_a)
    save_lightgbm_models(md_p15, md_p3, output_dir='models_standalone')
    
    # 7. Train Model E (MLP)
    print("\n[7/9] Training Model E (MLP)...")
    me_p15, me_p3, me_cols = train_model_mlp(X_a, y_p15_a, y_p3_a)
    save_mlp_models(me_p15, me_p3, me_cols, output_dir='models_standalone')
    
    # 8. Train Transformer
    print("\n[8/9] Training Transformer...")
    mt_model, mt_scaler = train_model_transformer(values)
    save_transformer_models(mt_model, mt_scaler, output_dir='models_standalone')
    
    # 9. Quick Simulation (Model A only for sanity check)
    print("\n[9/9] Running Simulation (Model A based)...")
    try:
        sim_len = min(500, len(X_a))
        X_sim = X_a.iloc[-sim_len:]
        true_vals = y_x_a[-sim_len:]
        preds_p15 = ma_p15.predict_proba(X_sim)[:, 1]
        preds_p3 = ma_p3.predict_proba(X_sim)[:, 1]
        preds_x = ma_x.predict(X_sim)
        
        sim_df = pd.DataFrame({
            'true_val': true_vals,
            'p_1_5': preds_p15,
            'p_3': preds_p3,
            'pred_x': preds_x
        })
        
        run_simulation(sim_df, model_name="Model A (Standalone)")
        print("Simulation complete. (Not a strict out-of-sample backtest; uses model A only.)")
    except Exception as e:
        print(f"Simulation step failed: {e}")
    
    print("\n✅ Standalone Run Completed Successfully!")
    print("Models saved to 'models_standalone' directory.")

if __name__ == "__main__":
    main()
