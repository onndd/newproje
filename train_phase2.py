
import os
import sys
import pandas as pd
import numpy as np
import joblib

# Ensure project is in path
if '/content/newproje' not in sys.path:
    sys.path.append('/content/newproje')

from jetx_project.data_loader import load_data, add_target_columns
from jetx_project.features import extract_features_batch
from jetx_project.model_crash import train_crash_detector
from jetx_project.config import DB_PATH

def main():
    print("🚀 Starting Phase 2 Training: Crash Guard & Magic Features 🚀")
    
    # 1. Load Data
    print("\n1. Loading Data...")
    df = load_data(DB_PATH)
    print(f"Loaded {len(df)} records.")
    
    # 2. Add Targets (Method Centralization)
    print("\n2. Adding Targets (P1.5, P3.0, CRASH)...")
    df = add_target_columns(df)
    print("Targets added. Checking 'target_crash' distribution:")
    print(df['target_crash'].value_counts(normalize=True))
    
    # 3. Feature Engineering (Magic Features included automatically)
    print("\n3. Extracting Features (including Trap & Symmetry)...")
    # We need to drop targets from features input if extract_features_batch expects raw df
    # But extract_features_batch takes 'value' column and generates features.
    # It does not peek at future.
    df_features = extract_features_batch(df[['value']]) 
    
    # Merge features with targets
    # Align indices
    df_full = pd.concat([df, df_features], axis=1).dropna()
    print(f"Features ready. Shape: {df_full.shape}")
    
    # 4. Train Crash Guard
    print("\n4. Training Crash Guard (LightGBM)...")
    # Select features (exclude targets and non-feature cols)
    drop_cols = ['id', 'value', 'target_p15', 'target_p3', 'target_crash', 'timestamp']
    feature_cols = [c for c in df_full.columns if c not in drop_cols]
    
    X = df_full[feature_cols]
    y_crash = df_full['target_crash']
    
    model = train_crash_detector(X, y_crash)
    print("\n✅ Crash Guard Training Complete!")
    print("You can now run the Ensemble prediction, and it will use this model to VETO risky bets.")

if __name__ == "__main__":
    main()
