
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import joblib
import os
from .config import MODEL_DIR
import matplotlib.pyplot as plt
import seaborn as sns

def train_crash_detector(X, y_crash, model_name="Crash_Guard"):
    """
    Trains a LightGBM model specifically to detect 'Crash' events (Multiplier < 1.20).
    This model acts as a Safety Guard.
    
    Args:
        X: Feature matrix
        y_crash: Binary target (1 = Crash, 0 = Safe)
    """
    print(f"\n--- Training {model_name} (Safety Guard) ---")
    
    # 1. Stratified K-Fold
    # We use CV to ensure robustness
    skf = StratifiedKFold(n_splits=3, shuffle=False)
    
    params = {
        'objective': 'binary',
        'objective': 'binary',
        'metric': 'auc', # Changed to AUC to better monitor separation
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        # 'is_unbalance': True, # Replaced with manual heavy weighting
        'scale_pos_weight': 5.0 # FORCE the model to pay 5x attention to Crashes
    }
    
    models = []
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_crash)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_crash.iloc[train_idx], y_crash.iloc[val_idx]
        
        # Create Dataset
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        # Train
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)]
        )
        
        # Eval
        preds_proba = model.predict(X_val, num_iteration=model.best_iteration)
        preds_bin = (preds_proba > 0.30).astype(int) # Lowered threshold to wake up the guard
        
        prec = precision_score(y_val, preds_bin, zero_division=0)
        rec = recall_score(y_val, preds_bin, zero_division=0)
        acc = accuracy_score(y_val, preds_bin)
        
        print(f"Fold {fold+1}: Accuracy: {acc:.4f}, Precision (Crash): {prec:.4f}, Recall (Crash): {rec:.4f}")
        scores.append(prec)
        models.append(model)
        
    avg_prec = np.mean(scores)
    print(f"Average CV Precision: {avg_prec:.4f}")
    
    # Select best model (simplest approach: fit on all data or take last? For now, retrain on all)
    print("Retraining on FULL dataset...")
    dtrain_full = lgb.Dataset(X, label=y_crash)
    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=models[-1].best_iteration # Use iter from last fold
    )
    
    # Save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    save_path = os.path.join(MODEL_DIR, f'{model_name}.joblib')
    joblib.dump(final_model, save_path)
    print(f"Crash Guard saved to {save_path}")
    
    return final_model

def load_crash_detector(model_name="Crash_Guard"):
    """
    Loads the trained Crash Detector model.
    """
    path = os.path.join(MODEL_DIR, f'{model_name}.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Crash Guard model not found at {path}")
    return joblib.load(path)

def predict_crash(model, X):
    """
    Returns probability of CRASH.
    """
    return model.predict(X)
