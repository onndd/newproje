
import numpy as np
import joblib
import os
from hmmlearn import hmm

def train_hmm_model(values, n_components=3):
    """
    Trains a Gaussian HMM on the data.
    Uses Log transformation to handle the exponential nature of multipliers.
    """
    # Reshape and Log Transform
    # JetX multipliers are exponential. Gaussian HMM works better with Log-Normal data.
    values_log = np.log1p(values).reshape(-1, 1)
    
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100)
    model.fit(values_log)
    
    # Analyze states to map them to Cold/Normal/Hot
    means = model.means_.flatten()
    sorted_indices = np.argsort(means)
    
    # Map: Smallest Mean -> 0 (Cold), Medium -> 1 (Normal), Largest -> 2 (Hot)
    state_map = {original: mapped for mapped, original in enumerate(sorted_indices)}
    
    return model, state_map

def predict_hmm_state(model, state_map, recent_values):
    """
    Predicts the current hidden state based on recent history.
    Returns:
        state_idx: 0 (Cold), 1 (Normal), 2 (Hot)
    """
    if len(recent_values) < 10:
        return 1 # Default to Normal if not enough data
        
    X = recent_values.reshape(-1, 1)
    hidden_states = model.predict(X)
    
    # The current state is the last one in the sequence
    current_internal_state = hidden_states[-1]
    
    # Map to our semantic states (0=Cold, 2=Hot)
    mapped_state = state_map[current_internal_state]
    
    return mapped_state

def save_hmm_model(model, state_map, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    joblib.dump({'model': model, 'map': state_map}, os.path.join(output_dir, 'model_hmm.pkl'))
    print(f"HMM model saved to {output_dir}")

def load_hmm_model(model_dir='.'):
    data = joblib.load(os.path.join(model_dir, 'model_hmm.pkl'))
    return data['model'], data['map']
