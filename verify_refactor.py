import numpy as np
import pandas as pd
import sqlite3
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath('.'))

def test_shape_mismatch_fix():
    print("\n--- Testing Shape Mismatch Fix (ensemble.py) ---")
    from jetx_project.ensemble import prepare_meta_features
    
    n_samples = 10
    preds_a = np.random.rand(n_samples)
    preds_b = np.random.rand(n_samples)
    preds_c = np.random.rand(n_samples)
    preds_d = np.random.rand(n_samples)
    preds_e = np.random.rand(n_samples)
    hmm_states = np.random.randint(0, 3, n_samples)
    
    # Test with None transformer
    try:
        meta_features = prepare_meta_features(
            preds_a, preds_b, preds_c, preds_d, preds_e, hmm_states, 
            values=None, preds_transformer=None
        )
        print(f"Meta features shape: {meta_features.shape}")
        # Expected columns: A, B, C, D, E, Transformer(Dummy), HMM(3), BustFreq(1) = 5 + 1 + 3 + 1 = 10
        expected_cols = 10
        if meta_features.shape[1] == expected_cols:
            print("✅ Shape mismatch fix PASSED: Dummy column added.")
        else:
            print(f"❌ Shape mismatch fix FAILED: Expected {expected_cols} columns, got {meta_features.shape[1]}")
            
        # Verify dummy values are 0.5
        # Transformer is the 6th column (index 5)
        if np.all(meta_features[:, 5] == 0.5):
             print("✅ Dummy values are correctly set to 0.5.")
        else:
             print("❌ Dummy values are NOT 0.5.")
             
    except Exception as e:
        print(f"❌ Shape mismatch fix FAILED with error: {e}")

def test_normalization_fix():
    print("\n--- Testing Normalization Fix (model_b.py) ---")
    from jetx_project.model_b import create_pattern_vector
    
    # Create a window with a huge value
    window = np.array([1.0, 2.0, 5000.0, 100000.0])
    # We need a longer window for the function usually, but let's see if we can test the logic directly
    # create_pattern_vector expects a full array and an index.
    
    # Let's mock the internal logic or just call it with a dummy array
    # create_pattern_vector(values, end_index, length=300)
    # We need at least 300 items.
    values = np.ones(300)
    values[-1] = 100000.0 # Huge value at the end
    
    pat = create_pattern_vector(values, 299, length=300)
    
    if pat is not None:
        # The first 300 elements are the normalized window
        norm_window = pat[:300]
        max_val = np.max(norm_window)
        min_val = np.min(norm_window)
        
        print(f"Max normalized value: {max_val}")
        print(f"Min normalized value: {min_val}")
        
        if max_val <= 1.0 and min_val >= 0.0:
            print("✅ Normalization fix PASSED: Values clipped to [0, 1].")
        else:
            print(f"❌ Normalization fix FAILED: Values out of range [{min_val}, {max_val}]")
    else:
        print("❌ Normalization fix FAILED: Pattern creation returned None.")

def test_db_reliability():
    print("\n--- Testing DB Reliability (data_loader.py) ---")
    from jetx_project.data_loader import load_data
    
    db_path = "test_empty.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    # Create empty DB file (no tables)
    conn = sqlite3.connect(db_path)
    conn.close()
    
    try:
        df = load_data(db_path)
        print(f"Loaded DataFrame from empty DB. Shape: {df.shape}")
        if df.empty and list(df.columns) == ['id', 'value']:
            print("✅ DB Reliability PASSED: Returned empty DataFrame for missing table.")
        else:
            print("❌ DB Reliability FAILED: Did not return expected empty DataFrame.")
    except Exception as e:
        print(f"❌ DB Reliability FAILED with error: {e}")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    test_shape_mismatch_fix()
    test_normalization_fix()
    test_db_reliability()
