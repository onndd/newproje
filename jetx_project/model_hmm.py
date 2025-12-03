
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
    # We switch to GMMHMM (Gaussian Mixture) to better model the heavy tails and complex distribution.
    # Zero-Inflated Logic: Filter out 1.00x (Instant Crashes)
    # These are "Dirac Delta" spikes at 1.0 that distort the continuous distribution.
    # We train HMM only on "Real Games" (e.g. > 1.01).
    mask_real = values > 1.01
    values_filtered = values[mask_real]
    
    print(f"HMM Training: Filtered {len(values) - len(values_filtered)} instant crashes (1.00x). Training on {len(values_filtered)} samples.")
    
    if len(values_filtered) < 100:
        print("Warning: Not enough data after filtering! Using all data.")
        values_filtered = values

    values_log = np.log1p(values_filtered).reshape(-1, 1)
    
    # GMMHMM with 5 components per state allows for more flexible density modeling
    # e.g. a state can have a "peak" and a "tail"
    model = hmm.GMMHMM(n_components=n_components, n_mix=5, covariance_type="full", n_iter=100, random_state=42)
    model.fit(values_log)
    
    # Analyze states to map them to Cold/Normal/Hot
    # For GMMHMM, means_ is (n_components, n_mix, n_features)
    # We can take the weighted average mean of mixtures for each component to sort them
    
    # weights: (n_components, n_mix)
    # means: (n_components, n_mix, 1)
    
    # Calculate effective mean for each state
    state_means = []
    for i in range(n_components):
        w = model.weights_[i]
        m = model.means_[i].flatten()
        effective_mean = np.dot(w, m)
        state_means.append(effective_mean)
        
    state_means = np.array(state_means)
    sorted_indices = np.argsort(state_means)
    
    # Map: Smallest Mean -> 0 (Cold), Medium -> 1 (Normal), Largest -> 2 (Hot)
    state_map = {original: mapped for mapped, original in enumerate(sorted_indices)}
    
    return model, state_map

def predict_hmm_state(model, values, state_map):
    """
    Predicts the state for a sequence of values using a pre-trained model.
    Returns the mapped state for EACH value in the sequence.
    CRITICAL: This function does NOT refit the model. It uses the provided model.
    """
    # Log Transform (Critical!)
    values_log = np.log1p(values).reshape(-1, 1)
    
    # Note: Even though we trained only on >1.01, we predict for ALL values.
    # 1.00x values will likely be assigned to the "Cold" state (lowest mean), which is correct behavior.
    
    hidden_states = model.predict(values_log)
    
    # Map all states
    mapped_states = np.array([state_map[s] for s in hidden_states])
    
    return mapped_states

def predict_hmm_states_causal(model, values, state_map, window_size=50):
    """
    Predicts HMM states causally using a rolling window.
    For each time t, we predict using data [t-window_size : t+1].
    This prevents future data leakage (Viterbi algorithm lookahead).
    """
    n = len(values)
    causal_states = np.zeros(n, dtype=int)
    
    # Log transform all values once
    values_log = np.log1p(values).reshape(-1, 1)
    
    print(f"Predicting HMM states causally (Window: {window_size})...")
    
    # For the first few items < window_size, we just predict on what we have
    # For items >= window_size, we use the window
    
    # Optimization: We can't easily vectorize this because 'predict' is complex.
    # We have to loop. 15k iterations is fine for training once.
    
    for i in range(n):
        start_idx = max(0, i - window_size + 1)
        # Slice up to i+1 (inclusive of i)
        window = values_log[start_idx : i + 1]
        
        if len(window) < 1:
            causal_states[i] = 0 # Default
            continue
            
        # Predict on the window
        # The last state in the sequence corresponds to time i
        try:
            hidden_states = model.predict(window)
            last_state = hidden_states[-1]
            causal_states[i] = state_map[last_state]
        except:
            causal_states[i] = 0 # Fallback
            
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i}/{n} samples...")
            
    return causal_states

def save_hmm_model(model, state_map, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    joblib.dump({'model': model, 'map': state_map}, os.path.join(output_dir, 'model_hmm.pkl'))
    print(f"HMM model saved to {output_dir}")

def load_hmm_model(model_dir='.'):
    data = joblib.load(os.path.join(model_dir, 'model_hmm.pkl'))
    return data['model'], data['map']
