import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

def calculate_profit_score(y_true, y_pred):
    """
    Calculates a custom score based on estimated profit.
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return -1000.0
        
    tn, fp, fn, tp = cm.ravel()
    
    # Custom Score Formula (Sniper Logic):
    # + TP * 100 (Big Reward for Risk - Incentive to Enter)
    # + TN * 1   (Minimal Reward for Safety - Prevent TN Farming)
    # - FP * 500 (DEATH PENALTY - Absolute Safety Requirement)
    # - FN * 20  (FOMO Penalty - Don't be too coward)
    # + Precision * 100 (Reliability Bonus)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    score = (tp * 100) + (tn * 1) - (fp * 500) - (fn * 20) + (precision * 100)
    return score

def find_best_threshold(y_true, y_prob, model_name, verbose=True, scoring_params=None):
    """
    Finds the optimal threshold based on Profit Scoring.
    """
    if scoring_params is None:
        from .config import PROFIT_SCORING_WEIGHTS
        scoring_params = PROFIT_SCORING_WEIGHTS
        
    best_thresh = 0.5
    best_score = -float('inf')
    
    # Coarse scan for speed during optimization (0.50 to 0.95 step 0.05)
    # Fine-tuning happens in final training
    thresholds = np.arange(0.05, 1.0, 0.05)
    
    for thresh in thresholds:
        preds = (y_prob > thresh).astype(int)
        
        # Handle cases where confusion_matrix might not be 2x2 (e.g., all predictions are same)
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        except ValueError:
            # If only one class is present in predictions, adjust
            if len(np.unique(preds)) == 1:
                if preds[0] == 0: # All predicted 0
                    tn = np.sum(y_true == 0)
                    fp = 0
                    fn = np.sum(y_true == 1)
                    tp = 0
                else: # All predicted 1
                    tn = 0
                    fp = np.sum(y_true == 0)
                    fn = 0
                    tp = np.sum(y_true == 1)
            else: # Should not happen if y_true has both classes
                continue # Skip this threshold if confusion matrix is weird
        
        score = (tp * scoring_params['TP']) - \
                (fp * scoring_params['FP']) + \
                (tn * scoring_params['TN']) - \
                (fn * scoring_params['FN'])
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
            
    return best_thresh, best_score

def optimize_catboost(X, y, n_trials=20, scoring_params=None, timeout=600):
    """
    Optimizes CatBoost hyperparameters using Optuna with GPU support.
    """
    print(f"--- Starting CatBoost Optimization ({n_trials} trials) ---")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    def objective(trial):
        # Class Weight Multiplier Optimization
        cw_multiplier = trial.suggest_float('cw_multiplier', 1.0, 5.0)
        class_weights = {0: cw_multiplier, 1: 1.0}

        param = {
            'iterations': trial.suggest_int('iterations', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'class_weights': class_weights,
            'od_type': 'Iter',
            'od_wait': 50,
            'task_type': 'GPU',
            'devices': '0',
            'verbose': 0
        }
        
        try:
            model = CatBoostClassifier(**param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
            
            preds_proba = model.predict_proba(X_val)[:, 1]
            # Custom Profit Metric
            best_thresh, best_score = find_best_threshold(y_val, preds_proba, "CatBoost_Opt", verbose=False, scoring_params=scoring_params)
            return best_score
                
        except Exception as e:
            print(f"GPU Error in trial: {e}")
            return -1000.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print(f"Best CatBoost params: {study.best_params}")
    return study.best_params

def optimize_lightgbm(X, y, n_trials=20, scoring_params=None, timeout=600):
    """
    Optimizes LightGBM hyperparameters using Optuna.
    """
    print(f"--- Starting LightGBM Optimization ({n_trials} trials) ---")
    
    # Split data (Time-series split)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    def objective(trial):
        param = {
            'objective': 'binary',
            'metric': 'binary_error',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            # Let Optuna decide if/how to weight
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
        }
        
        try:
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            # Train with sklearn API for consistency with predict_proba
            model = lgb.LGBMClassifier(**param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='logloss', callbacks=[lgb.early_stopping(100, verbose=False)])
            
            preds_proba = model.predict_proba(X_val)[:, 1]
            best_thresh, best_score = find_best_threshold(y_val, preds_proba, "LightGBM_Opt", verbose=False, scoring_params=scoring_params)
            return best_score
            
        except Exception as e:
            return -1000.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print(f"Best LightGBM params: {study.best_params}")
    return study.best_params



def optimize_mlp(X, y, n_trials=20, scoring_params=None, timeout=300):
    """
    Optimizes MLP hyperparameters.
    """
    print(f"--- Starting MLP Optimization ({n_trials} trials) ---")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    def objective(trial):
        # Oversampling Ratio
        os_ratio = trial.suggest_float('os_ratio', 1.0, 3.0)
        
        # Manual Oversampling for Minority (Class 0)
        from jetx_project.model_mlp import balance_data
        X_t_bal, y_t_bal = balance_data(X_train, y_train, target_class=0, multiplier=os_ratio)
        
        # Architecture
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_l{i}', 32, 256))
        
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(layers),
            activation='relu',
            solver='adam',
            alpha=trial.suggest_float('alpha', 0.0001, 0.1, log=True),
            learning_rate_init=trial.suggest_float('lr', 0.0001, 0.01, log=True),
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        
        clf.fit(X_t_bal, y_t_bal)
        
        preds = clf.predict(X_val)
        return calculate_profit_score(y_val, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print(f"Best MLP params: {study.best_params}")
    return study.best_params

def optimize_lstm(input_data, arg2=None, arg3=None, arg4=None, n_trials=10, scoring_params=None, timeout=600, target_threshold=1.50):
    """
    Optimizes LSTM hyperparameters.
    Supports TWO signatures for backward compatibility:
    1. New: optimize_lstm(values, n_trials=..., target_threshold=...)
    2. Old: optimize_lstm(X_train, y_train, X_val, y_val, n_trials=...)
    """
    print(f"--- Starting LSTM Optimization ({n_trials} trials, Target > {target_threshold}x) ---")
    
    # Determine mode
    if arg2 is not None and arg3 is not None and arg4 is not None:
        # Legacy Mode: (X_train, y_train, X_val, y_val) passed
        print("Legacy mode detected: Using provided X/y splits.")
        X_train, y_train, X_val, y_val = input_data, arg2, arg3, arg4
    else:
        # New Mode: Raw values passed
        print("New mode detected: Generating sequences from values.")
        values = input_data
        
        # Create Sequences (Internal)
        SEQ_LEN = 50 
        
        # 1. Split Raw Data FIRST (Leakage Fix)
        # Note: We need enough data for sequences.
        # We split 'values' directly.
        split_idx = int(len(values) * 0.85)
        train_values = values[:split_idx]
        val_values = values[split_idx:]
        
        # 2. Scale (Fit only on TRAIN)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        train_log = np.log1p(train_values)
        val_log = np.log1p(val_values)
        
        train_scaled = scaler.fit_transform(train_log.reshape(-1, 1))
        val_scaled = scaler.transform(val_log.reshape(-1, 1))
        
        # 3. Create Sequences
        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i : i + seq_len])
                y.append(data[i + seq_len])
            return np.array(X), np.array(y)
            
        X_train, y_train_scaled = create_sequences(train_scaled, SEQ_LEN)
        X_val, y_val_scaled = create_sequences(val_scaled, SEQ_LEN)
        
        # Target Conversion (Scaled -> Binary)
        # We need "Real" values to determine binary target > target_threshold
        # Inverse transform y
        y_train_real = np.expm1(scaler.inverse_transform(y_train_scaled))
        y_val_real = np.expm1(scaler.inverse_transform(y_val_scaled))
        
        y_train = (y_train_real >= target_threshold).astype(int).flatten()
        y_val = (y_val_real >= target_threshold).astype(int).flatten()
        
    def objective(trial):
        seq_len = X_train.shape[1]
        
        model = Sequential()
        
        # Layer 1
        units1 = trial.suggest_int('units1', 32, 256)
        dropout1 = trial.suggest_float('dropout1', 0.1, 0.5)
        model.add(LSTM(units1, return_sequences=True, input_shape=(seq_len, 1)))
        model.add(Dropout(dropout1))
        model.add(BatchNormalization())
        
        # Layer 2
        units2 = trial.suggest_int('units2', 32, 128)
        dropout2 = trial.suggest_float('dropout2', 0.1, 0.5)
        model.add(LSTM(units2, return_sequences=False))
        model.add(Dropout(dropout2))
        model.add(BatchNormalization())
        
        # Dense
        dense_units = trial.suggest_int('dense_units', 32, 128)
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        
        # Train
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10, # Short epochs for optimization
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Score
        preds_proba = model.predict(X_val, verbose=0).flatten()
        # Find best threshold dynamically
        best_thresh, best_score = find_best_threshold(y_val, preds_proba, "LSTM_Opt", verbose=False, scoring_params=scoring_params)
        return best_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print(f"Best LSTM params: {study.best_params}")
    
    # Return single dict. Caller handles duplication if needed.
    return study.best_params
```
