import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def prepare_meta_features(preds_a, preds_b, preds_c, preds_d, preds_e, hmm_states):
    """
    Combines predictions from all models into a single feature matrix for the meta-learner.
    
    Args:
        preds_a: Predictions from Model A (CatBoost)
        preds_b: Predictions from Model B (k-NN)
        preds_c: Predictions from Model C (LSTM)
        preds_d: Predictions from Model D (LightGBM)
        preds_e: Predictions from Model E (MLP)
        hmm_states: HMM States (Categorical)
        
    Returns:
        meta_features: Numpy array of shape (n_samples, n_features)
    """
    # Ensure all inputs are numpy arrays and have same length
    n_samples = len(preds_a)
    
    # One-Hot Encode HMM States (Assuming 3 states: 0, 1, 2)
    hmm_onehot = np.zeros((n_samples, 3))
    for i in range(n_samples):
        state = int(hmm_states[i])
        if 0 <= state < 3:
            hmm_onehot[i, state] = 1
            
    # Stack features
    # We use probabilities from each model + HMM context
    
    # NEW: Add "Recent 1.00x Frequency" (Instant Bust Risk)
    # We need to calculate this from raw values if provided, or pass it in.
    # To keep signature simple, let's assume 'values' is passed or we calculate it outside.
    # Actually, changing signature might break callers.
    # Let's add 'recent_busts' as an argument.
    
    # Wait, I need to update the signature in the definition.
    # But I can't see the caller here. The caller is in the notebook.
    # I will update the signature to accept 'recent_busts' (optional for backward compat, but we will use it).
    
    # Let's assume 'recent_busts' is passed.
    # If not passed, we use 0.
    
    meta_features_list = [
        preds_a,
        preds_b,
        preds_c,
        preds_d,
        preds_e,
        hmm_onehot
    ]
    
    # Check if we have extra args (hacky but effective without changing all signatures immediately)
    # Better: Update signature.
    
    return np.column_stack(meta_features_list)

def prepare_meta_features_v2(preds_a, preds_b, preds_c, preds_d, preds_e, hmm_states, values=None):
    """
    Enhanced version with 1.00x frequency feature.
    """
    n_samples = len(preds_a)
    
    # One-Hot Encode HMM
    hmm_onehot = np.zeros((n_samples, 3))
    for i in range(n_samples):
        state = int(hmm_states[i])
        if 0 <= state < 3:
            hmm_onehot[i, state] = 1
            
    # Calculate 1.00x Frequency (Last 50 games)
    bust_freq = np.zeros(n_samples)
    if values is not None:
        # values should be aligned such that values[i] is the game result at time i.
        # We want frequency of 1.00x in [i-50 : i]
        # Vectorized calculation
        is_bust = (values <= 1.01).astype(int) # 1.00 or 1.01
        # Rolling sum
        # We need pandas for easy rolling, or convolution
        import pandas as pd
        s_bust = pd.Series(is_bust)
        # shift(1) because we want past 50 games BEFORE current prediction
        bust_freq = s_bust.rolling(50).mean().shift(1).fillna(0).values
        
        # We need to slice bust_freq to match n_samples
        # Assuming values corresponds to the same timeline as preds
        if len(bust_freq) > n_samples:
            # Take the last n_samples
            bust_freq = bust_freq[-n_samples:]
        elif len(bust_freq) < n_samples:
            # Pad with 0
            bust_freq = np.pad(bust_freq, (n_samples - len(bust_freq), 0), 'constant')
            
    meta_features = np.column_stack([
        preds_a,
        preds_b,
        preds_c,
        preds_d,
        preds_e,
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
