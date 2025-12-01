
import numpy as np
import pandas as pd

import os
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    print("CatBoost not installed. Please install it via 'pip install catboost'")

from .features import extract_features

def prepare_model_a_data(values, hmm_states, start_index=500):
    """
    Prepares X (features) and y (targets) for Model A.
    """
    if len(hmm_states) != len(values):
        print(f"Warning: HMM States length ({len(hmm_states)}) != Values length ({len(values)}). Truncating to minimum.")
        min_len = min(len(hmm_states), len(values))
        values = values[:min_len]
        hmm_states = hmm_states[:min_len]

    X_list = []
    y_p15_list = []
    y_p3_list = []
    y_x_list = []
    
    # We go up to len(values) - 2 because we need target at i+1
    for i in range(start_index, len(values) - 1):
        # Extract features using history up to i
        feats = extract_features(values, i)
        
        # Add HMM State as a feature
        # hmm_states[i] is the state at time i
        feats['hmm_state'] = hmm_states[i]
        
        X_list.append(feats)
        
        # Target: Next value
        target_val = values[i+1]
        y_x_list.append(target_val)
        y_p15_list.append(1 if target_val >= 1.5 else 0)
        y_p3_list.append(1 if target_val >= 3.0 else 0)
        
    X = pd.DataFrame(X_list)
    return X, np.array(y_p15_list), np.array(y_p3_list), np.array(y_x_list)

def train_model_a(X_train, y_p15_train, y_p3_train, y_x_train):
    """
    Trains the 3 CatBoost models with validation and metric reporting.
    Uses 15% of the training data for validation to prevent overfitting.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

    # Split internal validation set (last 15% of training data to respect time order)
    split_idx = int(len(X_train) * 0.85)
    
    X_t, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    
    # 1. Model P1.5 (Classifier)
    print("\n--- Training Model A (P1.5) ---")
    y_p15_t, y_p15_val = y_p15_train[:split_idx], y_p15_train[split_idx:]
    
    model_p15 = CatBoostClassifier(
        iterations=2000, 
        learning_rate=0.03, 
        depth=6, # Reduced depth for CPU
        l2_leaf_reg=3,
        border_count=128,
        early_stopping_rounds=200, 
        eval_metric='AUC',
        verbose=100
        # Removed GPU requirement for compatibility
    )
    model_p15.fit(X_t, y_p15_t, eval_set=(X_val, y_p15_val))
    
    # Evaluate
    preds_p15 = model_p15.predict(X_val)
    acc_p15 = accuracy_score(y_p15_val, preds_p15)
    print(f"Validation Accuracy (P1.5): {acc_p15:.4f}")

    # 2. Model P3 (Classifier)
    print("\n--- Training Model A (P3.0) ---")
    y_p3_t, y_p3_val = y_p3_train[:split_idx], y_p3_train[split_idx:]
    
    model_p3 = CatBoostClassifier(
        iterations=2000, 
        learning_rate=0.03, 
        depth=6,
        l2_leaf_reg=3,
        border_count=128,
        early_stopping_rounds=200, 
        eval_metric='AUC',
        verbose=100
    )
    model_p3.fit(X_t, y_p3_t, eval_set=(X_val, y_p3_val))
    
    # Evaluate
    preds_p3 = model_p3.predict(X_val)
    acc_p3 = accuracy_score(y_p3_val, preds_p3)
    print(f"Validation Accuracy (P3.0): {acc_p3:.4f}")

    # 3. Model X (Regressor)
    print("\n--- Training Model A (Regression) ---")
    y_x_t, y_x_val = y_x_train[:split_idx], y_x_train[split_idx:]
    
    model_x = CatBoostRegressor(
        iterations=2000, 
        learning_rate=0.03, 
        depth=6,
        l2_leaf_reg=3,
        border_count=128,
        early_stopping_rounds=200, 
        loss_function='RMSE',
        verbose=100
    )
    model_x.fit(X_t, y_x_t, eval_set=(X_val, y_x_val))
    
    # Evaluate
    preds_x = model_x.predict(X_val)
    mse_x = mean_squared_error(y_x_val, preds_x)
    print(f"Validation MSE (X): {mse_x:.4f}")
    
    return model_p15, model_p3, model_x

def save_models(model_p15, model_p3, model_x, output_dir='.'):
    """
    Saves the trained models to disk.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_p15.save_model(os.path.join(output_dir, 'modelA_p15'))
    model_p3.save_model(os.path.join(output_dir, 'modelA_p3'))
    model_x.save_model(os.path.join(output_dir, 'modelA_x'))
    print(f"Models saved to {output_dir}")

def load_models(model_dir='.'):
    """
    Loads the trained models from disk.
    """
    model_p15 = CatBoostClassifier()
    model_p15.load_model(os.path.join(model_dir, 'modelA_p15'))
    
    model_p3 = CatBoostClassifier()
    model_p3.load_model(os.path.join(model_dir, 'modelA_p3'))
    
    model_x = CatBoostRegressor()
    model_x.load_model(os.path.join(model_dir, 'modelA_x'))
    
    return model_p15, model_p3, model_x
