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
    if values is not None:
        # Optimization: If we only need predictions for the last 'n_samples', 
        # and 'values' is huge (history), we don't need to roll over the entire history.
        # We need at least 50 (window) + n_samples (targets) + 1 (shift buffer)
        needed_len = n_samples + 55
        
        if len(values) > needed_len:
            # Slice to keep only relevant history
            values_slice = values[-needed_len:]
        else:
            values_slice = values
            
        # Vectorized calculation on slice
        is_bust = (values_slice <= 1.01).astype(int)
        
        import pandas as pd
        s_bust = pd.Series(is_bust)
        # shift(1) because we want past 50 games BEFORE current prediction
        bust_freq_slice = s_bust.rolling(50).mean().shift(1).fillna(0).values
        
        # Take the last n_samples
        if len(bust_freq_slice) >= n_samples:
            bust_freq = bust_freq_slice[-n_samples:]
        else:
            # Should not happen if logic is correct, but fallback
             bust_freq = np.pad(bust_freq_slice, (n_samples - len(bust_freq_slice), 0), 'constant')
            
    # Handle Transformer Predictions
    if preds_transformer is None:
        # If not provided, assume 0.5 (neutral) or 0? 
        # If the model was trained WITH transformer, this must be provided.
        # If trained WITHOUT, this column shouldn't exist.
        # For backward compatibility, let's assume we are moving to a new version where it exists.
        # We'll fill with 0.5 if missing, but ideally it should be passed.
        preds_transformer = np.full(n_samples, 0.5)
        
    meta_features = np.column_stack([
        preds_a,
        preds_b,
        preds_c,
        preds_d,
        preds_e,
        preds_transformer, # Added Transformer
        hmm_onehot,
        bust_freq
    ])
    
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
