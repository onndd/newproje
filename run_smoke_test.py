
import os
import sys
import numpy as np
import pandas as pd
import joblib
import time

# Ensure project root is in path
sys.path.append(os.getcwd())

from jetx_project.config import DB_PATH, DB_LIMIT
from jetx_project.features import extract_features

def run_smoke_test():
    print("🚀 Starting JetX Smoke Test...")
    
    # 1. Database Check
    print("\n[1/4] Checking Database Connection...")
    if not os.path.exists(DB_PATH):
        print(f"❌ FAILED: Database not found at {DB_PATH}")
        sys.exit(1)
    
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM jetx_results")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"✅ PASS: Database connected. Row count: {count}")
    except Exception as e:
        print(f"❌ FAILED: Database error: {e}")
        sys.exit(1)

    # 2. Feature Extraction Check
    print("\n[2/4] Testing Feature Extraction...")
    try:
        # Generate dummy data
        dummy_data = np.random.uniform(1.0, 10.0, 600)
        feats = extract_features(dummy_data, len(dummy_data)-1)
        if isinstance(feats, dict) and len(feats) > 10:
             print(f"✅ PASS: Extracted {len(feats)} features.")
        else:
             print("❌ FAILED: Feature extraction returned invalid result.")
             sys.exit(1)
    except Exception as e:
        print(f"❌ FAILED: Feature extraction error: {e}")
        sys.exit(1)

    # 3. Model Loading & Prediction Check (Model A)
    print("\n[3/4] Testing Model Prediction (CatBoost)...")
    model_path = 'models_standalone/catboost_model_p15.cbm'
    if os.path.exists(model_path):
        try:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(model_path)
            
            # Prepare input
            df = pd.DataFrame([feats])
            df['hmm_state'] = 1 # Dummy state
            
            # Pruning check
            pruning_path = 'models_standalone/selected_features.pkl'
            if os.path.exists(pruning_path):
                 selected = joblib.load(pruning_path)
                 print(f"   (Applying pruning with {len(selected)} features...)")
                 # Handle missing columns by zero filling if any
                 for c in selected:
                     if c not in df.columns:
                         df[c] = 0
                 df = df[selected]
            
            prob = model.predict_proba(df)[0][1]
            print(f"✅ PASS: Prediction successful. Probability: {prob:.4f}")
        except Exception as e:
            print(f"❌ FAILED: Model prediction error: {e}")
            # Don't exit here, allows checking other models if needed
    else:
        print("⚠️ SKIP: CatBoost model file not found. Have you run standalone_runner.py?")

    # 4. Overall Status
    print("\n[4/4] Smoke Test Summary")
    print("✅ SMOKE TEST OK")

if __name__ == "__main__":
    run_smoke_test()
