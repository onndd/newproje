
import numpy as np
import joblib
import os
from hmmlearn import hmm

def train_hmm_model(values, n_components=3):
    """
    Trains a Gaussian HMM to detect hidden states (regimes) of the game.
    States could be: 
    0: Low Volatility / Harvesting (Low multipliers)
    1: Normal
    2: High Volatility / Feeding (High multipliers)
    """
    # Reshape for HMM
    X = values.reshape(-1, 1)
    
    # We use GaussianHMM because multipliers are continuous
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
    model.fit(X)
    
    # We need to identify which state is which.
    # Usually, we sort states by their mean value.
    # State with lowest mean = Cold/Harvest
    # State with highest mean = Hot/Feeding
    
    means = model.means_.flatten()
    sorted_indices = np.argsort(means)
    
    # Create a mapping: 0 -> Cold, 1 -> Normal, 2 -> Hot
    # The model's internal state 0 might be Hot, so we map it.
    state_map = {
        original_idx: sorted_idx 
        for sorted_idx, original_idx in enumerate(sorted_indices)
    }
    
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
