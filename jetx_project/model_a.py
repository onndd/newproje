
import numpy as np
import pandas as pd

import os
from catboost import CatBoostClassifier, CatBoostRegressor

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

    # Convert to DataFrame for batch processing
    df = pd.DataFrame({'value': values})
    
    # Use Vectorized Feature Extraction (Much Faster)
    from .features import extract_features_batch
    X = extract_features_batch(df)
    
    # Prevent target leakage: remove raw value column from features
    if 'value' in X.columns:
        X = X.drop(columns=['value'])
    
    # Add HMM State
    X['hmm_state'] = hmm_states
    
    # Create Targets (Shifted by -1 because we predict next value)
    # Target for row i is value[i+1]
    # So we shift values by -1 to align "Next Value" with "Current Features"
    y_x_series = df['value'].shift(-1)
    y_p15_series = (y_x_series >= 1.5).astype(int)
    y_p3_series = (y_x_series >= 3.0).astype(int)
    
    # Filter valid range
    # We need start_index to avoid NaNs from rolling windows (usually 500)
    # And we need to drop the last row because it has no target (NaN after shift)
    
    valid_mask = (X.index >= start_index) & (X.index < len(values) - 1)
    
    X = X[valid_mask]
    y_p15 = y_p15_series[valid_mask].values
    y_p3 = y_p3_series[valid_mask].values
    y_x = y_x_series[valid_mask].values
    
    return X, y_p15, y_p3, y_x

def train_model_a(X_train, y_p15_train, y_p3_train, y_x_train, params_p15=None, params_p3=None, params_x=None):
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
    
    # Optimized parameters (Manual tuning to prevent overfitting)
    # Default params
    params = {
        'iterations': 2000,
        'learning_rate': 0.005, # Slower learning
        'depth': 6, # Reduced from 10 to 6
        'l2_leaf_reg': 9, # Increased regularization
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': 100,
        'early_stopping_rounds': 300,
        # Sınıf 0 ağırlığını artırdık çünkü model "her şeye giriyor". 
        # Negatifleri (0) daha ciddiye alması lazım.
        'class_weights': {0: 2.0, 1: 1.0},
    }
    
    # Override if params_p15 provided (from Optuna)
    if params_p15:
        print(f"Using optimized parameters for P1.5: {params_p15}")
        params.update(params_p15)

    model_p15 = CatBoostClassifier(**params)
    model_p15.fit(X_t, y_p15_t, eval_set=(X_val, y_p15_val))
    
    # Evaluate
    preds_p15 = model_p15.predict(X_val)
    acc_p15 = accuracy_score(y_p15_val, preds_p15)
    print(f"Validation Accuracy (P1.5): {acc_p15:.4f}")
    
    # Detailed Reporting
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_p15_val, preds_p15)
    print(f"Confusion Matrix (P1.5):\n{cm}")
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"Correctly Predicted >1.5x: {tp}/{tp+fn} (Recall: {tp/(tp+fn):.2%})")
        print(f"False Alarms: {fp}/{tp+fp} (Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.2%})")
    print("Classification Report:\n", classification_report(y_p15_val, preds_p15))
    
    # Feature Importance Analysis
    print("\nTop 10 Features (P1.5):")
    feature_importance = model_p15.get_feature_importance()
    feature_names = X_train.columns
    sorted_idx = np.argsort(feature_importance)[::-1]
    for i in range(min(10, len(feature_names))):
        idx = sorted_idx[i]
        print(f"{feature_names[idx]}: {feature_importance[idx]:.4f}")

    # 2. Model P3 (Classifier)
    print("\n--- Training Model A (P3.0) ---")
    y_p3_t, y_p3_val = y_p3_train[:split_idx], y_p3_train[split_idx:]
    # Optimized parameters (Manual tuning to prevent overfitting)
    # Default params
    params = {
        'iterations': 3000, 
        'learning_rate': 0.01, 
        'depth': 8, 
        'l2_leaf_reg': 5, 
        'loss_function': 'Logloss',
        'eval_metric': 'Precision', # Focus on correctness of positive predictions
        'auto_class_weights': 'Balanced', # Handle class imbalance
        'random_seed': 42,
        'verbose': 100,
        'early_stopping_rounds': 200 
    }
    
    # Override from Optuna
    if params_p3:
        print(f"Using optimized parameters for P3.0: {params_p3}")
        params.update(params_p3)

    model_p3 = CatBoostClassifier(**params)
    model_p3.fit(X_t, y_p3_t, eval_set=(X_val, y_p3_val))
    
    # Evaluate
    preds_p3 = model_p3.predict(X_val)
    acc_p3 = accuracy_score(y_p3_val, preds_p3)
    print(f"Validation Accuracy (P3.0): {acc_p3:.4f}")
    
    # Detailed Reporting
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_p3_val, preds_p3)
    print(f"Confusion Matrix (P3.0):\n{cm}")
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"Correctly Predicted >3.0x: {tp}/{tp+fn} (Recall: {tp/(tp+fn):.2%})")
        print(f"False Alarms: {fp}/{tp+fp} (Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.2%})")
    print("Classification Report:\n", classification_report(y_p3_val, preds_p3))

    # 3. Model X (Regressor)
    print("\n--- Training Model A (Regression) ---")
    y_x_t, y_x_val = y_x_train[:split_idx], y_x_train[split_idx:]
    
    
    # Defaults
    params = {
        'iterations': 1000, 
        'learning_rate': 0.03, 
        'depth': 6,
        'l2_leaf_reg': 3,
        'border_count': 128,
        'early_stopping_rounds': 100, 
        'loss_function': 'RMSE',
        'verbose': 100
    }
    
    if params_x:
        print(f"Using optimized parameters for Regression: {params_x}")
        params.update(params_x)

    model_x = CatBoostRegressor(**params)
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
    # Note: load_model should restore parameters saved during training.
    # If version mismatch occurs, ensure environment matches training (catboost==1.2.8).
    model_p15.load_model(os.path.join(model_dir, 'modelA_p15'))
    
    model_p3 = CatBoostClassifier()
    model_p3.load_model(os.path.join(model_dir, 'modelA_p3'))
    
    model_x = CatBoostRegressor()
    model_x.load_model(os.path.join(model_dir, 'modelA_x'))
    
    return model_p15, model_p3, model_x
