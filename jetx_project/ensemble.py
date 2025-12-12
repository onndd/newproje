import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

from .model_anomaly import check_anomaly

def prepare_meta_features(preds_a, preds_b, preds_c, preds_d, preds_e, hmm_states, values=None, preds_transformer=None, n_samples=None, n_hmm_components=3):
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
        hmm_states: Array of HMM states
        values: (Optional) Raw values for frequency calc
        preds_transformer: (Optional) Transformer preds
        n_samples: (Optional) Explicit sample count
        n_hmm_components: (Optional) Number of HMM states (default 3)
        
    Returns:
        meta_features: Numpy array of shape (n_samples, n_features)
    """
    # 1. Determine n_samples safely from PREDICTIONS (any available model)
    # The app might pass values (history=250) but only 1 prediction.
    if n_samples is None:
        # Check all prediction arrays for a valid length source
        available_preds = [arr for arr in [preds_a, preds_b, preds_c, preds_d, preds_e, preds_transformer] if arr is not None and len(arr) > 0]
        
        if not available_preds:
            # If NO models have predictions, we can't build meta-features unless purely creating empty dummy rows?
            # But usually this means something is wrong or we extract 0 features.
            # If 'values' is provided, maybe we infer from that? But usually we align to PREDICTIONS.
            if values is not None and len(values) > 0:
                 # Fallback: if we are just testing feature extraction without models?
                 # Rare case. Let's error if we strictly need predictions.
                 # But to be robust, let's look at history.
                 # Actually, usually getting here means system is deeply broken or just starting.
                 # Let's raise informative error ONLY if we really can't determine n_samples.
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
                # Audit Fix: Strict Error for length mismatch
                # Silent padding caused drift. We must fail fast.
                raise ValueError(f"CRITICAL: Meta-feature '{name}' length mismatch! Expected {n_samples}, got {len(arr)}. Alignment error in pipeline.")
            else:
                cleaned_inputs[name] = arr

    # HMM States handling - Align to n_samples
    if hmm_states is None or len(hmm_states) == 0:
        cleaned_hmm = np.full(n_samples, 1) # Default Normal
    else:
        # If hmm_states comes from full history (len=250) and n_samples=1
        if len(hmm_states) > n_samples:
            cleaned_hmm = hmm_states[-n_samples:]
        elif len(hmm_states) < n_samples:
            raise ValueError(f"CRITICAL: HMM States length mismatch! Expected {n_samples}, got {len(hmm_states)}.")
        else:
            cleaned_hmm = hmm_states
    
    # One-Hot Encode HMM States (Dynamic)
    hmm_onehot = np.zeros((n_samples, n_hmm_components))
    for i in range(n_samples):
        val = cleaned_hmm[i]
        try:
            state = int(val)
        except:
            state = 1 # Default Normal
            print(f"Warning: Invalid HMM state val '{val}'. Defaulting to 1.")

        # Strict Validation regarding n_hmm_components
        if state >= n_hmm_components:
             print(f"CRITICAL WARNING: Out-of-bound HMM state '{state}' (Max index: {n_hmm_components-1}). This implies model/config mismatch.")
             # Fallback to middle state to avoid crash, but log loudly
             state = n_hmm_components // 2
        
        if 0 <= state < n_hmm_components:
            hmm_onehot[i, state] = 1
            
    # Calculate 1.00x Frequency
    # We need to calculate this based on the *history available at each prediction point*.
    # If n_samples=1, we just calculate it for the last window.
    bust_freq = np.zeros(n_samples)
    
    if values is not None and len(values) > 0:
        import pandas as pd
        vals_array = np.array(values)
        
        # Logic: 
        # If n_samples == 1: Calculate freq on values[-50:]
        # If n_samples == len(values): Calculate rolling freq on all values
        
        is_bust = (vals_array <= 1.00).astype(int)
        bust_freq_series = pd.Series(is_bust).rolling(window=50, min_periods=1).mean()
        all_freqs = bust_freq_series.values
        
        if len(all_freqs) >= n_samples:
            bust_freq = all_freqs[-n_samples:]
        else:
            # Pad is okay here because it's feature extraction from history, not model prediction alignment
            pad_len = n_samples - len(all_freqs)
            bust_freq = np.pad(all_freqs, (pad_len, 0), mode='edge')
    else:
        pass
            
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
