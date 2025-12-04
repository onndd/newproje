import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
    n_samples = len(preds_a)
    
    inputs = {
        "preds_a": preds_a,
        "preds_b": preds_b,
        "preds_c": preds_c,
        "preds_d": preds_d,
        "preds_e": preds_e,
        "hmm_states": hmm_states,
    }
    for name, arr in inputs.items():
        if len(arr) != n_samples:
            raise ValueError(f"Length mismatch in meta features: expected {n_samples}, got {len(arr)} for {name}")
    
    # One-Hot Encode HMM States (Assuming 3 states: 0, 1, 2)
    hmm_onehot = np.zeros((n_samples, 3))
    for i in range(n_samples):
        state = int(hmm_states[i])
        if 0 <= state < 3:
            hmm_onehot[i, state] = 1
        else:
            print(f"Warning: Unknown HMM state {state} encountered. One-hot vector will be all zeros.")
            
    # Calculate 1.00x Frequency (Last 50 games)
    bust_freq = np.zeros(n_samples)
    # 1.00x Frequency Feature (Last 50 games)
    if values is not None and len(values) > 0:
        import pandas as pd
        is_bust = (values <= 1.00).astype(int)
        # Fix: Use min_periods=1 to handle short history (start of game)
        bust_freq_series = pd.Series(is_bust).rolling(window=50, min_periods=1).mean()
        
        # Align with prediction indices
        # If we are predicting for index N, we need stats from N-1
        # But here we are given 'values' which might be the full history or a slice.
        # We assume 'values' corresponds to the rows we are predicting for?
        # Actually, usually 'values' is the FULL history.
        # If so, we need to slice it to match n_samples.
        # However, the current implementation assumes 'values' aligns with the predictions somehow.
        # Let's stick to the original logic but make it robust.
        
        if len(bust_freq_series) >= n_samples:
             bust_freq_slice = bust_freq_series.iloc[-n_samples:].values
             bust_freq = bust_freq_slice
        else:
             # Pad with mean or 0 if not enough data
             current_vals = bust_freq_series.values
             pad_width = n_samples - len(current_vals)
             bust_freq = np.pad(current_vals, (pad_width, 0), 'edge') # Pad with last known value
    else:
        bust_freq = np.zeros(n_samples)
            
    # Handle Transformer Predictions
    # Fix: Only add transformer column if it is NOT None.
    # This prevents shape mismatch if the meta-learner was trained without it.
    
    feature_list = [
        preds_a,
        preds_b,
        preds_c,
        preds_d,
        preds_e
    ]
    
    if preds_transformer is not None:
        if len(preds_transformer) != n_samples:
            raise ValueError(f"Length mismatch in meta features: expected {n_samples}, got {len(preds_transformer)} for preds_transformer")
        feature_list.append(preds_transformer)
        
    feature_list.append(hmm_onehot)
    feature_list.append(bust_freq)

    meta_features = np.column_stack(feature_list)
    
    return meta_features

def train_meta_learner(meta_features, y_true):
    """
    Trains a Logistic Regression meta-learner.
    """
    # Scale features (important for Logistic Regression)
    scaler = StandardScaler()
    meta_features_scaled = scaler.fit_transform(meta_features)
    
    # Logistic Regression with L2 regularization
    # We want calibrated probabilities, so we use Logistic Regression
    meta_model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', random_state=42)
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
