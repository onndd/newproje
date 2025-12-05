
import lightgbm as lgb
import numpy as np
import os
from sklearn.metrics import accuracy_score

def train_model_lightgbm(X_train, y_p15_train, y_p3_train):
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
    clf_p15 = lgb.LGBMClassifier(
        n_estimators=2000, 
        learning_rate=0.01, 
        num_leaves=50, # Increased capacity
        min_child_samples=10, # Allow learning from fewer samples
        subsample=0.8,
        colsample_bytree=0.8
    )
    clf_p15.fit(X_t, y_p15_t, eval_set=[(X_val, y_p15_val)], eval_metric='logloss', 
                callbacks=[lgb.early_stopping(100)])
    
    # P3.0 Model
    print("Training LightGBM (P3.0)...")
    clf_p3 = lgb.LGBMClassifier(
        n_estimators=2000, 
        learning_rate=0.01, 
        num_leaves=50, 
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced' # Handle class imbalance
    )
    clf_p3.fit(X_t, y_p3_t, eval_set=[(X_val, y_p3_val)], eval_metric='logloss',
               callbacks=[lgb.early_stopping(100)])
               
    # Detailed Reporting
    from sklearn.metrics import confusion_matrix, classification_report
    
    # P1.5 Report
    preds_p15 = clf_p15.predict(X_val)
    print("\n--- LightGBM P1.5 Report ---")
    cm_p15 = confusion_matrix(y_p15_val, preds_p15)
    print(f"Confusion Matrix (P1.5):\n{cm_p15}")
    detailed_evaluation(y_p15_val, preds_p15, "P1.5", threshold=0.75)

    # P3.0 Report
    preds_p3 = clf_p3.predict(X_val)
    detailed_evaluation(y_p3_val, preds_p3, "P3.0", threshold=0.75)
               
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
