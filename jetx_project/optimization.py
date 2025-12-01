import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def optimize_catboost(X, y, n_trials=20, timeout=600):
    """
    Optimizes CatBoost hyperparameters using Optuna with GPU support.
    """
    print(f"--- Starting CatBoost Optimization ({n_trials} trials) ---")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    def objective(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'od_type': 'Iter',
            'od_wait': 50,
            'task_type': 'GPU', # Enable GPU
            'devices': '0',     # Use first GPU
            'verbose': 0
        }
        
        # Handle potential GPU errors gracefully (fallback to CPU if needed, but we want GPU)
        try:
            model = CatBoostClassifier(**param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
            preds = model.predict(X_val)
            accuracy = accuracy_score(y_val, preds)
            return accuracy
        except Exception as e:
            print(f"GPU Error in trial: {e}")
            return 0.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print(f"Best CatBoost params: {study.best_params}")
    return study.best_params

def optimize_lightgbm(X, y, n_trials=20, timeout=600):
    """
    Optimizes LightGBM hyperparameters using Optuna.
    """
    print(f"--- Starting LightGBM Optimization ({n_trials} trials) ---")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    def objective(trial):
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000)
        }
        
        # LightGBM GPU support requires special build, often safer to run on CPU or check support
        # We'll stick to CPU for LightGBM to avoid build issues, as CatBoost is the main heavy lifter
        
        model = lgb.LGBMClassifier(**param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print(f"Best LightGBM params: {study.best_params}")
    return study.best_params
