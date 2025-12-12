
import pytest
import numpy as np
import pandas as pd
import os
from jetx_project.model_a import prepare_model_a_data
from standalone_runner import load_data, get_values_array
from jetx_project.config import DB_LIMIT

# Mock data or load real data for testing
@pytest.fixture
def sample_data():
    """Fixture to provide sample data for tests."""
    # Create synthetic data if DB not available, or load small chunk
    values = np.random.lognormal(mean=0.5, sigma=0.5, size=600)
    # Clip to realistic range
    values = np.clip(values, 1.0, 100.0)
    return values

def test_feature_extraction(sample_data):
    """Test if feature extraction works and returns correct shape."""
    from jetx_project.features import extract_features
    
    current_idx = len(sample_data) - 1
    features = extract_features(sample_data, current_idx)
    
    assert isinstance(features, dict)
    assert len(features) > 0
    # Check for critical keys
    assert 'rolling_mean_10' in features
    assert 'volatility_50' in features

def test_hmm_loading_and_prediction(sample_data):
    """Test HMM model loading and state prediction."""
    from jetx_project.model_hmm import load_hmm_model, predict_categorical_hmm_states_causal
    
    model_dir = 'models_standalone'
    # Only run if model exists
    if not os.path.exists(os.path.join(model_dir, 'hmm_model.pkl')):
        pytest.skip("HMM model not found, skipping test.")
        
    hmm_model, hmm_map, hmm_bins = load_hmm_model(model_dir)
    assert hmm_model is not None
    
    # Predict states
    states = predict_categorical_hmm_states_causal(hmm_model, sample_data, hmm_map, bins=hmm_bins, window_size=200)
    assert len(states) == len(sample_data)
    assert set(states).issubset({0, 1, 2, 3, 4, 5}) # Assuming max states

def test_model_prediction_shape(sample_data):
    """Test if models produce valid probabilities (0-1)."""
    # Need features first
    from jetx_project.features import extract_features
    current_idx = len(sample_data) - 1
    feats = extract_features(sample_data, current_idx)
    df = pd.DataFrame([feats])
    
    # Add dummy hmm_state
    df['hmm_state'] = 1
    
    # Load Model A (CatBoost)
    import joblib
    model_path = 'models_standalone/catboost_model_p15.cbm'
    if not os.path.exists(model_path):
        pytest.skip("CatBoost model not found.")
        
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    # Predict
    prob = model.predict_proba(df)[0][1]
    assert isinstance(prob, (float, np.float32, np.float64))
    assert 0.0 <= prob <= 1.0
