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
    # 1. Determine n_samples dynamically from the first valid input
    n_samples = 0
    
    # Priority check for n_samples
    if values is not None:
        n_samples = len(values)
    elif preds_a is not None:
        n_samples = len(preds_a)
    elif preds_b is not None:
        n_samples = len(preds_b)
    # ... check others if needed, but usually one of above exists
    
    if n_samples == 0:
        # Extreme fallback
        return np.array([])
        
    inputs = {
        "preds_a": preds_a,
        "preds_b": preds_b,
        "preds_c": preds_c,
        "preds_d": preds_d,
        "preds_e": preds_e,
    }
    
    # Clean Inputs filling Nones
    cleaned_inputs = {}
    for name, arr in inputs.items():
        if arr is None or len(arr) == 0:
            print(f"Warning: {name} is missing. Filling with 0.5 (Neutral).")
            cleaned_inputs[name] = np.full(n_samples, 0.5)
        else:
            if len(arr) != n_samples:
                # Try to trim or pad? Better to fail or force fit?
                # For safety, strict length check but informative error
                if abs(len(arr) - n_samples) > 1: # slight tolerance
                     print(f"Critical Length Mismatch {name}: {len(arr)} vs {n_samples}. Truncating to min.")
                     min_l = min(len(arr), n_samples)
                     cleaned_inputs[name] = arr[:min_l]
                     # Only truncate others if this happens? 
                     # This is messy. Let's assume passed inputs are aligned or None.
                     # If just slightly different, we assume alignment.
                     pass 
                cleaned_inputs[name] = arr[:n_samples] # Truncate if longer
            else:
                cleaned_inputs[name] = arr

    # HMM States handling
    if hmm_states is None or len(hmm_states) == 0:
        print("Warning: hmm_states missing. Filling with 1 (Normal).")
        cleaned_hmm = np.full(n_samples, 1)
    else:
        cleaned_hmm = hmm_states[:n_samples]
    
    # One-Hot Encode HMM States (Assuming 3 states: 0, 1, 2)
    hmm_onehot = np.zeros((n_samples, 3))
    for i in range(n_samples):
        val = cleaned_hmm[i]
        try:
             state = int(val)
        except:
             state = 1 # fallback
             
        if 0 <= state < 3:
            hmm_onehot[i, state] = 1
        else:
            print(f"Warning: Unknown HMM state {state} encountered. One-hot vector will be all zeros.")
            
    # Calculate 1.00x Frequency (Fixed 50-game window on latest history)
    bust_freq = np.zeros(n_samples)
    if values is not None and len(values) > 0:
        import pandas as pd
        vals_array = np.array(values)
        window = max(50, n_samples)
        tail_vals = vals_array[-window:]
        is_bust = (tail_vals <= 1.00).astype(int)
        bust_freq_series = pd.Series(is_bust).rolling(window=50, min_periods=1).mean()
        tail_freq = bust_freq_series.values
        if len(tail_freq) >= n_samples:
            bust_freq = tail_freq[-n_samples:]
        else:
            pad_width = n_samples - len(tail_freq)
            bust_freq = np.pad(tail_freq, (pad_width, 0), mode='edge')
            
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
