
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

from .model_anomaly import check_anomaly
from .model_fourier import FourierDetector # New Import

def prepare_meta_features(preds_a, preds_b, preds_c, preds_d, preds_e, hmm_states, values=None, preds_transformer=None, n_samples=None, n_hmm_components=3):
    """
    Combines predictions from all models into a single feature matrix for the meta-learner.
    Includes 'Recent 1.00x Frequency' feature.
    Includes Transformer predictions.
    Includes Model G (Fourier) Rhythms.
    
    Args:
        preds_a: Predictions from Model A (CatBoost)
        preds_b: Predictions from Model B (k-NN)
        preds_c: Predictions from Model C (LSTM)
        preds_d: Predictions from Model D (LightGBM)
        preds_e: Predictions from Model E (MLP)
        hmm_states: Array of HMM states
        values: (Optional) Raw values for frequency & Fourier calc
        preds_transformer: (Optional) Transformer preds
        n_samples: (Optional) Explicit sample count
        n_hmm_components: (Optional) Number of HMM states (default 3)
        
    Returns:
        meta_features: Numpy array of shape (n_samples, n_features)
    """
    # 1. Determine n_samples safely from PREDICTIONS (any available model)
    if n_samples is None:
        available_preds = [arr for arr in [preds_a, preds_b, preds_c, preds_d, preds_e, preds_transformer] if arr is not None and len(arr) > 0]
        
        if not available_preds:
            if values is not None and len(values) > 0:
                 pass 
            raise ValueError("n_samples must be provided (or inferred from at least one model) to build meta-features.")
        
        n_samples = len(available_preds[0])
        
    inputs = {
        "preds_a": preds_a,
        "preds_b": preds_b,
        "preds_c": preds_c,
        "preds_d": preds_d,
        "preds_e": preds_e,
    }
    
    # Clean Inputs & strict Alignment
    cleaned_inputs = {}
    for name, arr in inputs.items():
        if arr is None or len(arr) == 0:
            cleaned_inputs[name] = np.full(n_samples, 0.5)
        else:
            if len(arr) != n_samples:
                raise ValueError(f"CRITICAL: Meta-feature '{name}' length mismatch! Expected {n_samples}, got {len(arr)}. Alignment error in pipeline.")
            else:
                cleaned_inputs[name] = arr

    # HMM States handling - Align to n_samples
    if hmm_states is None or len(hmm_states) == 0:
        cleaned_hmm = np.full(n_samples, 1) # Default Normal
    else:
        if len(hmm_states) > n_samples:
            cleaned_hmm = hmm_states[-n_samples:]
        elif len(hmm_states) < n_samples:
            raise ValueError(f"CRITICAL: HMM States length mismatch! Expected {n_samples}, got {len(hmm_states)}.")
        else:
            cleaned_hmm = hmm_states
    
    # One-Hot Encode HMM States
    hmm_onehot = np.zeros((n_samples, n_hmm_components))
    for i in range(n_samples):
        val = cleaned_hmm[i]
        try:
            state = int(val)
        except:
            state = 1
        
        if state >= n_hmm_components:
             state = n_hmm_components // 2
        
        if 0 <= state < n_hmm_components:
            hmm_onehot[i, state] = 1
            
    # Calculate 1.00x Frequency
    bust_freq = np.zeros(n_samples)
    
    if values is not None and len(values) > 0:
        vals_array = np.array(values)
        is_bust = (vals_array <= 1.00).astype(int)
        bust_freq_series = pd.Series(is_bust).rolling(window=50, min_periods=1).mean()
        all_freqs = bust_freq_series.values
        
        if len(all_freqs) >= n_samples:
            bust_freq = all_freqs[-n_samples:]
        else:
            pad_len = n_samples - len(all_freqs)
            bust_freq = np.pad(all_freqs, (pad_len, 0), mode='edge')

    # --- MODEL G: FOURIER ANALYSIS ---
    # We define the windows here as per spec
    fourier_detector = FourierDetector(window_sizes=[64, 256, 1024])
    
    # We need n_samples rows of features
    # Each row 'i' corresponds to the state at time 't'
    # Ideally, we calculate this on the full 'values' history and slice the last n_samples.
    
    # Default (empty) features if values missing
    fourier_features_list = []
    
    if values is not None and len(values) > 0:
        vals_array = np.array(values)
        # Run Batch Analysis
        # This returns a DataFrame with same length as vals_array
        fourier_df = fourier_detector.analyze_batch(vals_array)
        
        if len(fourier_df) >= n_samples:
            # Slice last n_samples
            df_slice = fourier_df.iloc[-n_samples:]
        else:
            # Pad
            pad_len = n_samples - len(fourier_df)
            # Create padding df with neutral values
            pad_data = {}
            for col in fourier_df.columns:
                if 'strength' in col: val = 0.0
                else: val = 0.5
                pad_data[col] = np.full(pad_len, val)
            pad_df = pd.DataFrame(pad_data)
            df_slice = pd.concat([pad_df, fourier_df], ignore_index=True)
            
        # Convert to list of arrays (columns)
        for col in df_slice.columns:
            fourier_features_list.append(df_slice[col].values)
    else:
        # Create zero/neutral arrays
        # 3 windows * 2 features = 6 columns
        for w in [64, 256, 1024]:
            fourier_features_list.append(np.zeros(n_samples)) # strength
            fourier_features_list.append(np.full(n_samples, 0.5)) # phase
            
    # Handle Transformer Predictions
    feature_list = [
        cleaned_inputs["preds_a"],
        cleaned_inputs["preds_b"],
        cleaned_inputs["preds_c"],
        cleaned_inputs["preds_d"],
        cleaned_inputs["preds_e"]
    ]
    
    if preds_transformer is not None:
        if len(preds_transformer) != n_samples:
            raise ValueError(f"Length mismatch in meta features: expected {n_samples}, got {len(preds_transformer)} for preds_transformer")
        feature_list.append(preds_transformer)
    else:
        dummy_transformer = np.full(n_samples, 0.5)
        feature_list.append(dummy_transformer)
        
    feature_list.append(hmm_onehot)
    feature_list.append(bust_freq)
    
    # Append Fourier Features (6 columns)
    feature_list.extend(fourier_features_list)

    meta_features = np.column_stack(feature_list)
    
    return meta_features

def train_meta_learner(meta_features, y_true):
    """
    Trains a CatBoost meta-learner (gradient boosting).
    """
    scaler = StandardScaler()
    meta_features_scaled = scaler.fit_transform(meta_features)
    
    print(f"Training Meta-Learner (CatBoost) on {meta_features.shape[1]} features...")
    meta_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=4,
        loss_function='Logloss',
        eval_metric='AUC',
        verbose=100,
        early_stopping_rounds=50,
        allow_writing_files=False,
        random_seed=42
    )
    meta_model.fit(meta_features_scaled, y_true)
    
    # Feature Importance Reporting (Added for Verification)
    try:
        importance = meta_model.get_feature_importance()
        # We don't have exact col names here easily, but we know indices.
        # Just simple print of top importance
        print("Meta-Learner Top Feature Importances (Indices):", np.argsort(importance)[::-1][:5])
    except:
        pass

    return meta_model, scaler

def save_meta_learner(model, scaler, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    joblib.dump({'model': model, 'scaler': scaler}, os.path.join(output_dir, 'meta_learner.pkl'))
    print(f"Meta-learner saved to {output_dir}")

def load_meta_learner(model_dir='.'):
    data = joblib.load(os.path.join(model_dir, 'meta_learner.pkl'))
    return data['model'], data['scaler']

def predict_meta(model, scaler, meta_features):
    """
    Predicts final probabilities using the meta-learner.
    """
    meta_features_scaled = scaler.transform(meta_features)
    return model.predict_proba(meta_features_scaled)[:, 1]

def predict_meta_safe(model, scaler, meta_features, anomaly_model=None, current_window_values=None, crash_preds=None):
    """
    Predicts using Meta-Learner with Circuit Breaker (Anomaly Detection) and Crash Guard (Veto).
    """
    # 1. Check Anomaly (Circuit Breaker)
    if anomaly_model is not None and current_window_values is not None:
        score, _ = check_anomaly(anomaly_model, current_window_values)
        if score == -1:
            print("CIRCUIT BREAKER: Anomaly Detected! Betting suspended.")
            # Return 0 probability to prevent betting
            return np.zeros(len(meta_features))
            
    # 2. Crash Guard Veto
    # If the Crash Model says "Safe Crash" (> 85%), we VETO the prediction.
    if crash_preds is not None:
        # Check if ANY prediction in the batch is unsafe (or handle vector-wise)
        # We assume crash_preds matches length of meta_features
        if len(crash_preds) != len(meta_features):
             print(f"WARNING: Crash preds length mismatch ({len(crash_preds)} vs {len(meta_features)}). Skipping Veto.")
        else:
            # Vectorized Veto
            # If crash_prob > 0.85, set meta_prob to 0
            # We calculate normal preds first
            raw_probs = predict_meta(model, scaler, meta_features)
            
            # Apply Veto
            # mask: 1 if crash likely, 0 otherwise
            veto_mask = (crash_preds > 0.85).astype(int)
            if np.sum(veto_mask) > 0:
                print(f"CRASH GUARD: Vetoed {np.sum(veto_mask)} bets due to high crash risk.")
                
            final_probs = raw_probs * (1 - veto_mask)
            return final_probs
            
    # 3. Normal Prediction (if no veto)
    return predict_meta(model, scaler, meta_features)
