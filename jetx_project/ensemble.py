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
    meta_features = np.column_stack([
        preds_a,
        preds_b,
        preds_c,
        preds_d,
        preds_e,
        hmm_onehot
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
