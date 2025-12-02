
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
    clf_p15 = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.01, num_leaves=31)
    clf_p15.fit(X_t, y_p15_t, eval_set=[(X_val, y_p15_val)], eval_metric='logloss', 
                callbacks=[lgb.early_stopping(100)])
    
    # P3.0 Model
    print("Training LightGBM (P3.0)...")
    clf_p3 = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.01, num_leaves=31)
    clf_p3.fit(X_t, y_p3_t, eval_set=[(X_val, y_p3_val)], eval_metric='logloss',
               callbacks=[lgb.early_stopping(100)])
               
    return clf_p15, clf_p3

def save_lightgbm_models(model_p15, model_p3, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_p15.booster_.save_model(os.path.join(output_dir, 'modelD_p15.txt'))
    model_p3.booster_.save_model(os.path.join(output_dir, 'modelD_p3.txt'))
    print(f"LightGBM models saved to {output_dir}")
