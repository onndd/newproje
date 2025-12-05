import os
import sqlite3
import pandas as pd
import sys
from unittest.mock import MagicMock

# Setup paths
sys.path.append(os.path.abspath('.'))
DB_PATH = 'jetx_test.db'
os.environ['DB_PATH'] = DB_PATH

def create_dummy_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE jetx_results (id INTEGER PRIMARY KEY AUTOINCREMENT, value REAL)")
    # Insert 3000 records
    data = [(float(i),) for i in range(3000)]
    cursor.executemany("INSERT INTO jetx_results (value) VALUES (?)", data)
    conn.commit()
    conn.close()
    print("Created dummy DB with 3000 records.")

def test_data_loader_limit():
    from jetx_project.data_loader import load_data
    df = load_data(DB_PATH, limit=2000)
    print(f"Loaded {len(df)} records with limit=2000.")
    assert len(df) == 2000, f"Expected 2000 records, got {len(df)}"
    # Check if we got the last 2000 (ids 1001 to 3000)
    # Since we insert 0 to 2999, values are 0.0 to 2999.0
    # Last 2000 should be 1000.0 to 2999.0
    assert df['value'].iloc[-1] == 2999.0, f"Expected last value 2999.0, got {df['value'].iloc[-1]}"
    assert df['value'].iloc[0] == 1000.0, f"Expected first value 1000.0, got {df['value'].iloc[0]}"
    print("Data loader limit test passed.")

def test_graceful_degradation():
    """
    Ensures load_all_models tolerates missing model files and returns a dict with None entries.
    """
    # Mock streamlit
    sys.modules['streamlit'] = MagicMock()
    import streamlit as st
    st.cache_resource = lambda func: func
    st.error = MagicMock()
    st.warning = MagicMock()
    st.success = MagicMock()
    st.title = MagicMock()
    st.set_page_config = MagicMock()
    st.number_input = MagicMock(return_value=1.0)
    st.button = MagicMock(return_value=False)
    st.write = MagicMock()
    st.metric = MagicMock()
    st.subheader = MagicMock()
    st.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
    class SessionState(dict):
        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc
        def __setattr__(self, key, value):
            self[key] = value
    st.session_state = SessionState()
    st.stop = MagicMock()
    
    # Mock model loading functions to fail
    mock_model_a = MagicMock()
    mock_model_a.load_models = MagicMock(side_effect=Exception("Model A missing"))
    mock_model_a.prepare_model_a_data = MagicMock()
    sys.modules['jetx_project.model_a'] = mock_model_a
    # Stub other model modules to avoid heavy deps during import
    mock_model_b = MagicMock()
    mock_model_b.load_memory = MagicMock(return_value=(None, None, None, None))
    mock_model_b.create_pattern_vector = MagicMock()
    mock_model_b.predict_model_b = MagicMock()
    sys.modules['jetx_project.model_b'] = mock_model_b
    
    mock_model_lstm = MagicMock()
    mock_model_lstm.load_lstm_models = MagicMock(return_value=(None, None, None))
    mock_model_lstm.create_sequences = MagicMock()
    sys.modules['jetx_project.model_lstm'] = mock_model_lstm
    
    mock_model_lightgbm = MagicMock()
    mock_model_lightgbm.load_lightgbm_models = MagicMock(return_value=(None, None))
    sys.modules['jetx_project.model_lightgbm'] = mock_model_lightgbm
    
    mock_model_mlp = MagicMock()
    mock_model_mlp.load_mlp_models = MagicMock(return_value=(None, None, None))
    sys.modules['jetx_project.model_mlp'] = mock_model_mlp
    
    mock_model_hmm = MagicMock()
    mock_model_hmm.load_hmm_model = MagicMock(return_value=(None, None, None))
    mock_model_hmm.predict_hmm_state = MagicMock()
    sys.modules['jetx_project.model_hmm'] = mock_model_hmm
    
    mock_ensemble = MagicMock()
    mock_ensemble.load_meta_learner = MagicMock(return_value=(None, None))
    mock_ensemble.prepare_meta_features = MagicMock()
    mock_ensemble.predict_meta = MagicMock()
    sys.modules['jetx_project.ensemble'] = mock_ensemble
    
    mock_transformer = MagicMock()
    mock_transformer.load_transformer_models = MagicMock(return_value=(None, None))
    sys.modules['jetx_project.model_transformer'] = mock_transformer
    
    import importlib
    app_module = importlib.import_module('app')
    
    models = app_module.load_all_models()
    
    assert isinstance(models, dict), "load_all_models should return a dict even on failures"
    assert 'model_a' in models, "models dict must contain model_a key"
    assert models['model_a'] is None, "model_a should degrade to None when load fails"

if __name__ == "__main__":
    create_dummy_db()
    test_data_loader_limit()
    # Clean up
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    print("All verification tests passed!")
