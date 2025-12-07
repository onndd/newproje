import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

from .model_anomaly import check_anomaly

def prepare_meta_features(preds_a, preds_b, preds_c, preds_d, preds_e, hmm_states, values=None, preds_transformer=None):
    """
    Combines predictions from all models into a single feature matrix for the meta-learner.
    Includes 'Recent 1.00x Frequency' feature if values are provided.
    Includes Transformer predictions if provided.
    
    Args:
        preds_a: Predictions from Model A (CatBoost)
        preds_b: Predictions from Model B (k-NN)
        preds_c: Predictions from Model C (LSTM)
        preds_d: Predictions from Model D (LightGBM)
        preds_e: Predictions from Model E (MLP)
        hmm_states: HMM States (Categorical)
        values: Raw game values (optional, required for 1.00x frequency feature)
        preds_transformer: Predictions from Transformer (optional)
        
    Returns:
        meta_features: Numpy array of shape (n_samples, n_features)
    """
    # 1. Determine n_samples safely from PREDICTIONS first (not values)
    # The app might pass values (history=250) but only 1 prediction.
    # We must trust the prediction arrays as the ground truth for "rows to predict".
    n_samples = 0
    
    # Priority check for n_samples from predictions
    if preds_a is not None and len(preds_a) > 0:
        n_samples = len(preds_a)
    elif preds_b is not None and len(preds_b) > 0:
        n_samples = len(preds_b)
    elif preds_c is not None and len(preds_c) > 0:
        n_samples = len(preds_c)
    elif preds_d is not None and len(preds_d) > 0:
        n_samples = len(preds_d)
    elif preds_e is not None and len(preds_e) > 0:
        n_samples = len(preds_e)
    # Only fall back to values length if NO predictions exist (unlikely but possible during init)
    elif values is not None:
        n_samples = len(values)
        
    if n_samples == 0:
        return np.array([])
        
    inputs = {
        "preds_a": preds_a,
        "preds_b": preds_b,
        "preds_c": preds_c,
        "preds_d": preds_d,
        "preds_e": preds_e,
    }
    
    # Clean Inputs filling Nones & Alignment
    cleaned_inputs = {}
    for name, arr in inputs.items():
        if arr is None or len(arr) == 0:
            cleaned_inputs[name] = np.full(n_samples, 0.5)
        else:
            if len(arr) != n_samples:
                # If array is longer (e.g. accidentally passed full history?), truncate to n_samples
                # But usually, it's safer to just take the last n_samples? 
                # Or if it's 1 vs 250, we assume the CALLER messed up.
                # Here we assume standard behavior: truncate if longer, pad/fill if shorter (rare/bad)
                if len(arr) > n_samples:
                    # Take LAST n_samples (assuming time series alignment)
                    cleaned_inputs[name] = arr[-n_samples:]
                else:
                    # Shorter? Fill with 0.5 for missing slots (or pad first/last)
                    # This is a critical error state usually, but let's be robust:
                    pad_len = n_samples - len(arr)
                    cleaned_inputs[name] = np.pad(arr, (pad_len, 0), constant_values=0.5)
            else:
                cleaned_inputs[name] = arr

    # HMM States handling - Align to n_samples
    if hmm_states is None or len(hmm_states) == 0:
        cleaned_hmm = np.full(n_samples, 1)
    else:
        # If hmm_states comes from full history (len=250) and n_samples=1
        if len(hmm_states) > n_samples:
            cleaned_hmm = hmm_states[-n_samples:]
        elif len(hmm_states) < n_samples:
            pad_len = n_samples - len(hmm_states)
            cleaned_hmm = np.pad(hmm_states, (pad_len, 0), constant_values=1)
        else:
            cleaned_hmm = hmm_states
    
    # One-Hot Encode HMM States
    hmm_onehot = np.zeros((n_samples, 3))
    for i in range(n_samples):
        val = cleaned_hmm[i]
        try:
             state = int(val)
        except:
             state = 1
        if 0 <= state < 3:
            hmm_onehot[i, state] = 1
            
    # Calculate 1.00x Frequency
    # We need to calculate this based on the *history available at each prediction point*.
    # If n_samples=1, we just calculate it for the last window.
    bust_freq = np.zeros(n_samples)
    
    if values is not None and len(values) > 0:
        import pandas as pd
        vals_array = np.array(values)
        
        # If we are predicting for n_samples rows, we essentially need rolling features for the last n_samples.
        # But 'values' might be the FULL history including the ones we are predicting for?
        # Usually 'values' passed here is the history context.
        
        # Logic: 
        # If n_samples == 1: Calculate freq on values[-50:]
        # If n_samples == len(values): Calculate rolling freq on all values
        
        # We'll compute full rolling metrics on 'values' then take the last n_samples
        is_bust = (vals_array <= 1.00).astype(int)
        bust_freq_series = pd.Series(is_bust).rolling(window=50, min_periods=1).mean()
        all_freqs = bust_freq_series.values
        
        if len(all_freqs) >= n_samples:
            bust_freq = all_freqs[-n_samples:]
        else:
            # Pad
            pad_len = n_samples - len(all_freqs)
            bust_freq = np.pad(all_freqs, (pad_len, 0), mode='edge')
    else:
        # No values provided? 0.0
        pass
            
    # Handle Transformer Predictions
    # Fix: Only add transformer column if it is NOT None.
    # This prevents shape mismatch if the meta-learner was trained without it.
    
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
        # Fix: Add dummy column (neutral 0.5) to maintain shape for Meta-Learner
        # The meta-learner was trained with this column, so it expects it.
        dummy_transformer = np.full(n_samples, 0.5)
        feature_list.append(dummy_transformer)
        
    feature_list.append(hmm_onehot)
    feature_list.append(bust_freq)

    meta_features = np.column_stack(feature_list)
    
    return meta_features

def train_meta_learner(meta_features, y_true):
    """
    Trains a CatBoost meta-learner (gradient boosting) instead of Logistic Regression.
    """
    # Özellikleri tutarlılık için ölçekliyoruz (CatBoost şart koşmuyor ama normalize ediyoruz)
    scaler = StandardScaler()
    meta_features_scaled = scaler.fit_transform(meta_features)
    
    print("Training Meta-Learner (CatBoost)...")
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

def predict_meta_safe(model, scaler, meta_features, anomaly_model=None, current_window_values=None):
    """
    Predicts using Meta-Learner with Circuit Breaker (Anomaly Detection).
    """
    # 1. Check Anomaly (Circuit Breaker)
    if anomaly_model is not None and current_window_values is not None:
        score, _ = check_anomaly(anomaly_model, current_window_values)
        if score == -1:
            print("CIRCUIT BREAKER: Anomaly Detected! Betting suspended.")
            # Return 0 probability to prevent betting
            return np.zeros(len(meta_features))
            
    # 2. Normal Prediction
    return predict_meta(model, scaler, meta_features)
