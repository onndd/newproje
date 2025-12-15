
import lightgbm as lgb
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from .evaluation import detailed_evaluation

def train_model_lightgbm(X_train, y_p15_train, y_p3_train, params_p15=None, params_p3=None, scoring_params_p15=None, scoring_params_p3=None):
    """
    Trains LightGBM models.
    """
    # Split
    split_idx = int(len(X_train) * 0.85)
    X_t, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_p15_t, y_p15_val = y_p15_train[:split_idx], y_p15_train[split_idx:]
    y_p3_t, y_p3_val = y_p3_train[:split_idx], y_p3_train[split_idx:]
    
    
    # Define Helper for Threshold Search
    from .config import PROFIT_SCORING_WEIGHTS, SCORING_LIGHTGBM
    def find_best_threshold(y_true, y_prob, model_name, verbose=True, scoring_params=None):
        """
        Finds the optimal threshold based on Profit Scoring.
        """
        if scoring_params is None:
            scoring_params = SCORING_LIGHTGBM
            
        best_thresh = 0.5
        best_score = -float('inf')
        thresholds = np.arange(0.50, 0.99, 0.01)
        
        if verbose:
            print(f"\nScanning Thresholds for {model_name}...")
            
        for thresh in thresholds:
            preds = (y_prob > thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            score = (tp * scoring_params['TP']) - \
                    (fp * scoring_params['FP']) + \
                    (tn * scoring_params['TN']) - \
                    (fn * scoring_params['FN'])
            
            if score > best_score:
                best_score = score
                best_thresh = thresh
        
        if verbose:
            print(f"Best Threshold for {model_name}: {best_thresh:.2f} (Score: {best_score})")
        return best_thresh, best_score

    # 1. Model P1.5 (Classifier)
    # -----------------------------------------------------
    print("Training LightGBM (P1.5)...")
    # Tuned parameters to fix underfitting
    params = {
        'n_estimators': 2000, 
        'learning_rate': 0.01, 
        'num_leaves': 50, # Increased capacity
        'min_child_samples': 10, # Allow learning from fewer samples
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        # 'class_weight': 'balanced' # Removed aggressive balancing to reduce FP
    }
    if params_p15:
        print(f"Using optimized parameters for LightGBM P1.5: {params_p15}")
        params.update(params_p15)
        
        # Ensure we don't pass 'class_weight' if it's None (Optuna might pass it)
        if 'class_weight' in params and params['class_weight'] is None:
            params.pop('class_weight') # Let LGBM handle spread naturally or use scale_pos_weight
        
    # --- ROLLING WINDOW CV (P1.5) ---
    print("\n[CV] Running 3-Fold Rolling Window CV for P1.5...")
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        cv_X_train, cv_X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        cv_y_train, cv_y_val = y_p15_train[train_idx], y_p15_train[val_idx]
        
        cv_model = lgb.LGBMClassifier(**params)
        cv_model.fit(cv_X_train, cv_y_train, eval_set=[(cv_X_val, cv_y_val)], eval_metric='logloss', 
                     callbacks=[lgb.early_stopping(100, verbose=False)])
        
        probs = cv_model.predict_proba(cv_X_val)[:, 1]
        _, score = find_best_threshold(cv_y_val, probs, f"P1.5 Fold {fold+1}", verbose=False)
        cv_scores.append(score)
        print(f"  Fold {fold+1} Profit Score: {score:.2f}")
        
    print(f"[CV] Average P1.5 Score: {np.mean(cv_scores):.2f}")
    # --------------------------------

    # -----------------------------------------------------
    # OPTIONAL: Feature Selection (Top 20)
    # -----------------------------------------------------
    # User requested to use only the Best 20 features to reduce noise.
    # We will use the P1.5 fit to determine importance (or we could do it separately).
    
    print("\n[Feature Selection] Selecting Top 20 Features...")
    
    # 1. Fit a strict temporary model to find generic importance
    selector_model = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
    selector_model.fit(X_t, y_p15_t)
    
    # 2. Get Importance
    importances = selector_model.feature_importances_
    feature_names = X_t.columns
    
    # 3. Select Top 20
    # Sort indices
    indices = np.argsort(importances)[::-1]
    top_20_indices = indices[:20]
    top_20_features = feature_names[top_20_indices].tolist()
    
    print(f"Top 20 Features Selected: {top_20_features}")
    
    # 4. Filter Datasets
    X_train_sel = X_train[top_20_features]
    X_val_sel = X_val[top_20_features]
    X_t_sel = X_t[top_20_features]
    
    # Re-define P1.5 Training with Selected Features
    print("Training LightGBM (P1.5) with Selected Features...")
    clf_p15 = lgb.LGBMClassifier(**params)
    clf_p15.fit(X_t_sel, y_p15_t, eval_set=[(X_val_sel, y_p15_val)], eval_metric='logloss', 
                callbacks=[lgb.early_stopping(100)])
    
    # 2. Model P3.0 (Classifier)
    # -----------------------------------------------------
    print("Training LightGBM (P3.0) with Selected Features...")
    params_3 = params.copy()
    if params_p3:
        print(f"Using optimized parameters for LightGBM P3.0: {params_p3}")
        params_3.update(params_p3)
        if 'class_weight' in params_3 and params_3['class_weight'] is None:
            params_3.pop('class_weight')
        
    # --- ROLLING WINDOW CV (P3.0) ---
    print("\n[CV] Running 3-Fold Rolling Window CV for P3.0...")
    cv_scores_p3 = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        cv_X_train, cv_X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        cv_y_train, cv_y_val = y_p3_train[train_idx], y_p3_train[val_idx]
        
        # Apply Feature Selection to CV
        cv_X_train_sel = cv_X_train[top_20_features]
        cv_X_val_sel = cv_X_val[top_20_features]
        
        cv_model = lgb.LGBMClassifier(**params_3)
        cv_model.fit(cv_X_train_sel, cv_y_train, eval_set=[(cv_X_val_sel, cv_y_val)], eval_metric='logloss', 
                     callbacks=[lgb.early_stopping(100, verbose=False)])
        
        probs = cv_model.predict_proba(cv_X_val_sel)[:, 1]
        _, score = find_best_threshold(cv_y_val, probs, f"P3.0 Fold {fold+1}", verbose=False)
        cv_scores_p3.append(score)
        print(f"  Fold {fold+1} Profit Score: {score:.2f}")
        
    print(f"[CV] Average P3.0 Score: {np.mean(cv_scores_p3):.2f}")
    # --------------------------------

    clf_p3 = lgb.LGBMClassifier(**params_3)
    clf_p3.fit(X_t_sel, y_p3_t, eval_set=[(X_val_sel, y_p3_val)], eval_metric='logloss',
               callbacks=[lgb.early_stopping(100)])
               
    # Detailed Reporting with Dynamic Thresholding
    
    # P1.5 Report
    print("\n--- LightGBM P1.5 Report ---")
    preds_p15_proba = clf_p15.predict_proba(X_val_sel)[:, 1]
    best_thresh_p15, best_score_p15 = find_best_threshold(y_p15_val, preds_p15_proba, "LightGBM P1.5", scoring_params=scoring_params_p15)
    
    # Use best threshold for reporting (Not detailed_evaluation call anymore)
    preds_p15 = (preds_p15_proba > best_thresh_p15).astype(int)
    cm_p15 = confusion_matrix(y_p15_val, preds_p15)
    print(f"Confusion Matrix (P1.5 @ {best_thresh_p15:.2f}):\n{cm_p15}")
    detailed_evaluation(y_p15_val, preds_p15_proba, "P1.5", threshold=best_thresh_p15)

    # P3.0 Report
    print("\n--- LightGBM P3.0 Report ---")
    preds_p3_proba = clf_p3.predict_proba(X_val_sel)[:, 1]
    best_thresh_p3, best_score_p3 = find_best_threshold(y_p3_val, preds_p3_proba, "LightGBM P3.0", scoring_params=scoring_params_p3)
    
    preds_p3 = (preds_p3_proba > best_thresh_p3).astype(int)
    cm_p3 = confusion_matrix(y_p3_val, preds_p3)
    print(f"Confusion Matrix (P3.0 @ {best_thresh_p3:.2f}):\n{cm_p3}")
    detailed_evaluation(y_p3_val, preds_p3_proba, "P3.0", threshold=best_thresh_p3)
               
    return clf_p15, clf_p3, top_20_features

def save_lightgbm_models(model_p15, model_p3, output_dir='.'):
    import joblib
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    joblib.dump(model_p15, os.path.join(output_dir, 'modelD_p15.pkl'))
    joblib.dump(model_p3, os.path.join(output_dir, 'modelD_p3.pkl'))
    print(f"LightGBM models saved to {output_dir}")

def load_lightgbm_models(model_dir='.'):
    import joblib
    p15_path = os.path.join(model_dir, 'modelD_p15.pkl')
    p3_path = os.path.join(model_dir, 'modelD_p3.pkl')
    
    if not os.path.exists(p15_path) or not os.path.exists(p3_path):
        return None, None
        
    model_p15 = joblib.load(p15_path)
    model_p3 = joblib.load(p3_path)
    
    return model_p15, model_p3
