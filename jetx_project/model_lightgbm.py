
import lightgbm as lgb
import numpy as np
import os
from sklearn.metrics import accuracy_score
from .evaluation import detailed_evaluation

def train_model_lightgbm(X_train, y_p15_train, y_p3_train, params_p15=None, params_p3=None):
    """
    Trains LightGBM models.
    """
    # Split
    split_idx = int(len(X_train) * 0.85)
    X_t, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_p15_t, y_p15_val = y_p15_train[:split_idx], y_p15_train[split_idx:]
    y_p3_t, y_p3_val = y_p3_train[:split_idx], y_p3_train[split_idx:]
    
    # P1.5 Model
    print("Training LightGBM (P1.5)...")
    # Tuned parameters to fix underfitting
    params = {
        'n_estimators': 2000, 
        'learning_rate': 0.01, 
        'num_leaves': 50, # Increased capacity
        'min_child_samples': 10, # Allow learning from fewer samples
        'subsample': 0.8,
        'colsample_bytree': 0.8,
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
        
    clf_p15 = lgb.LGBMClassifier(**params)
    clf_p15.fit(X_t, y_p15_t, eval_set=[(X_val, y_p15_val)], eval_metric='logloss', 
                callbacks=[lgb.early_stopping(100)])
    
    # P3.0 Model
    print("Training LightGBM (P3.0)...")
    params_3 = {
        'n_estimators': 2000, 
        'learning_rate': 0.01, 
        'num_leaves': 50, 
        'min_child_samples': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        # 'class_weight': 'balanced' # Removed mostly, let Optuna decide
    }
    if params_p3:
        print(f"Using optimized parameters for LightGBM P3.0: {params_p3}")
        params_3.update(params_p3)
        
        if 'class_weight' in params_3 and params_3['class_weight'] is None:
            params_3.pop('class_weight')
        
    clf_p3 = lgb.LGBMClassifier(**params_3)
    clf_p3.fit(X_t, y_p3_t, eval_set=[(X_val, y_p3_val)], eval_metric='logloss',
               callbacks=[lgb.early_stopping(100)])
               
    # Detailed Reporting with Dynamic Thresholding
    from sklearn.metrics import confusion_matrix
    from .config import PROFIT_SCORING_WEIGHTS
    
    def find_best_threshold(y_true, y_prob, model_name):
        best_thresh = 0.5
        best_score = -float('inf')
        
        # Scan thresholds
        thresholds = np.arange(0.50, 0.99, 0.01)
        
        print(f"\nScanning Thresholds for {model_name}...")
        for thresh in thresholds:
            preds = (y_prob > thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            
            # Profit Score (Sniper)
            score = (tp * PROFIT_SCORING_WEIGHTS['tp']) + \
                    (fp * PROFIT_SCORING_WEIGHTS['fp']) + \
                    (tn * PROFIT_SCORING_WEIGHTS['tn']) + \
                    (fn * PROFIT_SCORING_WEIGHTS['fn'])
            
            if score > best_score:
                best_score = score
                best_thresh = thresh
        
        print(f"Best Threshold for {model_name}: {best_thresh:.2f} (Score: {best_score})")
        return best_thresh

    # P1.5 Report
    print("\n--- LightGBM P1.5 Report ---")
    preds_p15_proba = clf_p15.predict_proba(X_val)[:, 1]
    best_thresh_p15 = find_best_threshold(y_p15_val, preds_p15_proba, "LightGBM P1.5")
    
    # Use best threshold for reporting (Not detailed_evaluation call anymore)
    from sklearn.metrics import classification_report
    preds_p15 = (preds_p15_proba > best_thresh_p15).astype(int)
    cm_p15 = confusion_matrix(y_p15_val, preds_p15)
    print(f"Confusion Matrix (P1.5 @ {best_thresh_p15:.2f}):\n{cm_p15}")
    detailed_evaluation(y_p15_val, preds_p15_proba, "P1.5", threshold=best_thresh_p15)

    # P3.0 Report
    print("\n--- LightGBM P3.0 Report ---")
    preds_p3_proba = clf_p3.predict_proba(X_val)[:, 1]
    best_thresh_p3 = find_best_threshold(y_p3_val, preds_p3_proba, "LightGBM P3.0")
    
    preds_p3 = (preds_p3_proba > best_thresh_p3).astype(int)
    cm_p3 = confusion_matrix(y_p3_val, preds_p3)
    print(f"Confusion Matrix (P3.0 @ {best_thresh_p3:.2f}):\n{cm_p3}")
    detailed_evaluation(y_p3_val, preds_p3_proba, "P3.0", threshold=best_thresh_p3)
               
    return clf_p15, clf_p3

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
