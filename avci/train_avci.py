
import pandas as pd
import numpy as np
import optuna
import joblib
import os
from .config_avci import DB_PATH, TARGETS, SCORING_1_5, SCORING_2_0, SCORING_3_0, SCORING_5_0, SCORING_10_0, SCORING_20_0, SCORING_50_0, SCORING_100_0
from .data_avci import load_data, add_targets
from .features_avci import extract_features
from .models_avci import train_lgbm, objective_lgbm

def get_scoring_params(target):
    if target == 1.5: return SCORING_1_5
    if target == 2.0: return SCORING_2_0
    if target == 3.0: return SCORING_3_0
    if target == 5.0: return SCORING_5_0
    if target == 10.0: return SCORING_10_0
    if target == 20.0: return SCORING_20_0
    if target == 50.0: return SCORING_50_0
    if target == 100.0: return SCORING_100_0
    return SCORING_3_0 # Default

def run_training(epochs=50 # Optuna Trials
                ):
    print("--- AVCI: Hunter Protocol Initiated ---")
    
    # 1. Load Data
    print("Loading Data...")
    df = load_data(DB_PATH, limit=100000)
    
    # 2. Features
    print("Extracting Features...")
    df = extract_features(df)
    
    # 3. Targets
    print("Labelling Targets...")
    df = add_targets(df, TARGETS)
    
    # 4. Split
    features = [c for c in df.columns if 'target' not in c and 'result' not in c and 'value' not in c and 'id' not in c]
    X = df[features]
    
    split_idx = int(len(df) * 0.85)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    
    os.makedirs('models', exist_ok=True)
    
    # 5. Train Loop for Each Target
    for target in TARGETS:
        print(f"\nTargetting: {target}x")
        y_col = f'target_{str(target).replace(".","_")}'
        y = df[y_col]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        scoring = get_scoring_params(target)
        
        # Optuna
        print(f"Optimizing for {target}x...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective_lgbm(trial, X_train, y_train, X_val, y_val, scoring), n_trials=epochs)
        
        print(f"Best Params for {target}x: {study.best_params}")
        print(f"Best Profit Score: {study.best_value}")
        
        # Final Train
        best_params = study.best_params
        best_params.update({'metric': 'binary_logloss', 'objective': 'binary', 'verbosity': -1})
        
        model = train_lgbm(X_train, y_train, X_val, y_val, best_params)
        model.save_model(f'models/avci_lgbm_{str(target).replace(".","_")}.txt')
        print(f"Model saved: models/avci_lgbm_{str(target).replace('.','_')}.txt")

if __name__ == "__main__":
    run_training()
