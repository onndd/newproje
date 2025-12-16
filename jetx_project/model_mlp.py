
from sklearn.neural_network import MLPClassifier
import joblib
import os
import numpy as np
import pandas as pd


def balance_data(X, y, target_class, multiplier=2.0):
    mask_target = y == target_class
    mask_other = y != target_class
    
    X_target, y_target = X[mask_target], y[mask_target]
    X_other, y_other = X[mask_other], y[mask_other]
    
    import math
    repeats = math.ceil(multiplier)
    
    X_target_balanced = pd.concat([X_target] * repeats, axis=0)
    y_target_balanced = np.tile(y_target, repeats)
    
    X_balanced = pd.concat([X_other, X_target_balanced], axis=0)
    y_balanced = np.concatenate([y_other, y_target_balanced])
    
    perm = np.random.permutation(len(X_balanced))
    return X_balanced.iloc[perm], y_balanced[perm]

def train_model_mlp(X_train, y_p15_train, y_p3_train, params_p15=None, params_p3=None, scoring_params_p15=None, scoring_params_p3=None):
    """
    Trains MLP models.
    """
    # Filter features for MLP: Use ONLY Raw Lags and HMM State
    # This forces the Neural Network to learn its own representations without our "hand-crafted" features.
    # We want diversity in the ensemble.
    
    feature_cols = [col for col in X_train.columns if col.startswith('raw_lag_') or col == 'hmm_state']
    X_train_filtered = X_train[feature_cols].copy()
    
    print(f"MLP Input Features: {len(feature_cols)} (Raw Lags + HMM Only)")
    
    # Handle NaNs for MLP
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_train_filtered)
    X_train_filtered = pd.DataFrame(X_imputed, columns=X_train_filtered.columns, index=X_train_filtered.index)
    
    # Scaling (Critical for MLP)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_filtered)
    X_train_filtered = pd.DataFrame(X_scaled, columns=X_train_filtered.columns, index=X_train_filtered.index)
    print("MLP Data Scaled (StandardScaler).")
    
    # Split
    split_idx = int(len(X_train_filtered) * 0.85)
    X_t, X_val = X_train_filtered.iloc[:split_idx], X_train_filtered.iloc[split_idx:]
    y_p15_t, y_p15_val = y_p15_train[:split_idx], y_p15_train[split_idx:]
    y_p3_t, y_p3_val = y_p3_train[:split_idx], y_p3_train[split_idx:]
    
    
    # Define Helper for Threshold Search
    from .config import PROFIT_SCORING_WEIGHTS, SCORING_MLP
    # Use centralized logic from optimization.py
    from .optimization import find_best_threshold

    # Compute Sample Weights for Class Balancing
    from sklearn.utils.class_weight import compute_sample_weight
    
    # -----------------------------------------------------
    # 1. Model P1.5 (Classifier)
    # -----------------------------------------------------
    # Manual Oversampling for Class Balancing (since MLPClassifier doesn't support class_weight)
    
    # P1.5 Model: Minority is Class 0 (Loss < 1.50) -> ~35%
    # We want to boost Class 0 to prevent "Always Yes"
    print("Training MLP (P1.5) - Balancing Minority (Class 0)...")
    
    params = {
        'hidden_layer_sizes': (256, 128, 64), 
        'activation': 'relu', 
        'solver': 'adam', 
        'alpha': 0.01, 
        'learning_rate_init': 0.001,
        'max_iter': 500, 
        'early_stopping': True, 
        'verbose': False # Silent for CV
    }
    target_multiplier = 2.0
    if params_p15:
        print(f"Using optimized parameters for MLP P1.5: {params_p15}")
        # Make a copy to avoid modifying the original dict if reusable
        params.update(params_p15.copy())
        
        # Extract os_ratio for data balancing
        if 'os_ratio' in params:
            target_multiplier = params['os_ratio']
            del params['os_ratio'] # Explicit delete
            
        # Construct hidden_layer_sizes if n_layers is present
        if 'n_layers' in params:
            n_layers = params['n_layers']
            del params['n_layers']
            layers = []
            for i in range(n_layers):
                key = f'n_units_l{i}'
                if key in params:
                    layers.append(params[key])
                    del params[key]
            if layers:
                params['hidden_layer_sizes'] = tuple(layers)
            
    # --- ROLLING WINDOW CV (P1.5) ---
    print("\n[CV] Running 3-Fold Rolling Window CV for MLP P1.5...")
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_filtered)):
        cv_X_train, cv_X_val = X_train_filtered.iloc[train_idx], X_train_filtered.iloc[val_idx]
        cv_y_train, cv_y_val = y_p15_train[train_idx], y_p15_train[val_idx]
        
        # Balance Data for Training Fold
        cv_X_train_bal, cv_y_train_bal = balance_data(cv_X_train, cv_y_train, target_class=0, multiplier=target_multiplier)
        
        cv_model = MLPClassifier(**params)
        cv_model.fit(cv_X_train_bal, cv_y_train_bal)
        
        probs = cv_model.predict_proba(cv_X_val)[:, 1]
        _, score = find_best_threshold(cv_y_val, probs, f"P1.5 Fold {fold+1}", verbose=False)
        cv_scores.append(score)
        print(f"  Fold {fold+1} Profit Score: {score:.2f}")
        
    print(f"[CV] Average P1.5 Score: {np.mean(cv_scores):.2f}")
    # --------------------------------

    params['verbose'] = True # Restore verbose
    X_t_p15, y_p15_t_balanced = balance_data(X_t, y_p15_t, target_class=0, multiplier=target_multiplier)
    clf_p15 = MLPClassifier(**params)
    clf_p15.fit(X_t_p15, y_p15_t_balanced)
    
    # -----------------------------------------------------
    # 2. Model P3.0 (Classifier)
    # -----------------------------------------------------
    print("Training MLP (P3.0)...")
    
    params_3 = {
        'hidden_layer_sizes': (256, 128, 64), 
        'activation': 'relu', 
        'solver': 'adam', 
        'alpha': 0.01, 
        'learning_rate_init': 0.001,
        'max_iter': 500, 
        'early_stopping': True, 
        'verbose': False
    }
    target_multiplier_3 = 2.0
    
    if params_p3:
        print(f"Using optimized parameters for MLP P3.0: {params_p3}")
        params_3.update(params_p3.copy())
        
        if 'os_ratio' in params_3:
            target_multiplier_3 = params_3['os_ratio']
            del params_3['os_ratio']
            
        if 'n_layers' in params_3:
            n_layers = params_3['n_layers']
            del params_3['n_layers']
            layers = []
            for i in range(n_layers):
                key = f'n_units_l{i}'
                if key in params_3:
                    layers.append(params_3[key])
                    del params_3[key]
            if layers:
                params_3['hidden_layer_sizes'] = tuple(layers)
                
    # --- ROLLING WINDOW CV (P3.0) ---
    print("\n[CV] Running 3-Fold Rolling Window CV for MLP P3.0...")
    cv_scores_p3 = []
    
    # Note: P3.0 usually has minority class 1 (High Win), so we balance target_class=1
    # Check default logic in balance_data usage for P3?
    # Actually P3 wins are rare (prob ~33% or less, actually values >3 is 33%, no wait. values >3.0 is 33% probability -> 1/3 odds)
    # 1.5x is 64%. 3.0x is 33%. 
    # So for P1.5, >1.5 (Class 1) is Majority. Class 0 is Minority. (We balanced 0). Correct.
    # For P3.0, >3.0 (Class 1) is Minority. Class 0 is Majority. (We should balance 1).
    # Let's check original code logic for P3.0 balancing... 
    # Original code didn't explicit loop for P3 logic in view, wait.
    # I need to be careful. Let's assume P3 balances Class 1.
    
    # In P3, Class 1 is Target (Rare). So we usually balance Class 1.
    target_cls_p3 = 1 
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_filtered)):
        cv_X_train, cv_X_val = X_train_filtered.iloc[train_idx], X_train_filtered.iloc[val_idx]
        cv_y_train, cv_y_val = y_p3_train[train_idx], y_p3_train[val_idx]
        
        # Balance Data for Training Fold -- target class 1 for P3
        cv_X_train_bal, cv_y_train_bal = balance_data(cv_X_train, cv_y_train, target_class=target_cls_p3, multiplier=target_multiplier_3)
        
        cv_model = MLPClassifier(**params_3)
        cv_model.fit(cv_X_train_bal, cv_y_train_bal)
        
        probs = cv_model.predict_proba(cv_X_val)[:, 1]
        _, score = find_best_threshold(cv_y_val, probs, f"P3.0 Fold {fold+1}", verbose=False)
        cv_scores_p3.append(score)
        print(f"  Fold {fold+1} Profit Score: {score:.2f}")
        
    print(f"[CV] Average P3.0 Score: {np.mean(cv_scores_p3):.2f}")
    # --------------------------------
    
    params_3['verbose'] = True
    X_t_p3, y_p3_t_balanced = balance_data(X_t, y_p3_t, target_class=target_cls_p3, multiplier=target_multiplier_3)
    clf_p3 = MLPClassifier(**params_3)
    clf_p3.fit(X_t_p3, y_p3_t_balanced)

    
    # Detailed Reporting with Dynamic Thresholding
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    # P1.5 Report
    print("\n--- MLP P1.5 Report ---")
    preds_p15_prob = clf_p15.predict_proba(X_val)[:, 1]
    best_thresh_p15, _ = find_best_threshold(y_p15_val, preds_p15_prob, "MLP P1.5", scoring_params=scoring_params_p15)
    
    preds_p15 = (preds_p15_prob > best_thresh_p15).astype(int)
    cm_p15 = confusion_matrix(y_p15_val, preds_p15)
    print(f"Confusion Matrix (P1.5 @ {best_thresh_p15:.2f}):\n{cm_p15}")
    if cm_p15.shape == (2, 2):
        tn, fp, fn, tp = cm_p15.ravel()
        print(f"Correctly Predicted >1.5x: {tp}/{tp+fn} (Recall: {tp/(tp+fn):.2%})")
        print(f"False Alarms: {fp}/{tp+fp} (Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.2%})")
    print(classification_report(y_p15_val, preds_p15))

    # P3.0 Report
    print("\n--- MLP P3.0 Report ---")
    preds_p3_prob = clf_p3.predict_proba(X_val)[:, 1]
    best_thresh_p3, _ = find_best_threshold(y_p3_val, preds_p3_prob, "MLP P3.0", scoring_params=scoring_params_p3)
    
    preds_p3 = (preds_p3_prob > best_thresh_p3).astype(int)
    cm_p3 = confusion_matrix(y_p3_val, preds_p3)
    print(f"Confusion Matrix (P3.0 @ {best_thresh_p3:.2f}):\n{cm_p3}")
    if cm_p3.shape == (2, 2):
        tn, fp, fn, tp = cm_p3.ravel()
        print(f"Correctly Predicted >3.0x: {tp}/{tp+fn} (Recall: {tp/(tp+fn):.2%})")
        print(f"False Alarms: {fp}/{tp+fp} (Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.2%})")
    print(classification_report(y_p3_val, preds_p3))
    
    return clf_p15, clf_p3, feature_cols, scaler # Return scaler to save/use during prediction

def save_mlp_models(model_p15, model_p3, feature_cols, scaler, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    joblib.dump(model_p15, os.path.join(output_dir, 'modelE_p15.pkl'))
    joblib.dump(model_p3, os.path.join(output_dir, 'modelE_p3.pkl'))
    joblib.dump(feature_cols, os.path.join(output_dir, 'modelE_cols.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'modelE_scaler.pkl'))
    print(f"MLP models saved to {output_dir}")

def load_mlp_models(model_dir='.'):
    p15_path = os.path.join(model_dir, 'modelE_p15.pkl')
    p3_path = os.path.join(model_dir, 'modelE_p3.pkl')
    cols_path = os.path.join(model_dir, 'modelE_cols.pkl')
    scaler_path = os.path.join(model_dir, 'modelE_scaler.pkl')
    
    if not os.path.exists(p15_path) or not os.path.exists(p3_path) or not os.path.exists(cols_path):
        return None, None, None, None
        
    model_p15 = joblib.load(p15_path)
    model_p3 = joblib.load(p3_path)
    feature_cols = joblib.load(cols_path)
    
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = None
    
    return model_p15, model_p3, feature_cols, scaler
