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
    
    # Custom Score Formula:
    # + TP * 10 (Win)
    # + TN * 5 (Save)
    # - FP * 50 (Loss - HUGE PENALTY)
    # - FN * 2 (Missed Opp)
    # + Precision * 100 (Reliability Bonus)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    score = (tp * 10) + (tn * 5) - (fp * 50) - (fn * 2) + (precision * 100)
    return score

def optimize_catboost(X, y, n_trials=20, timeout=600):
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
            threshold = 0.50
            preds = (preds_proba >= threshold).astype(int)
            
            return calculate_profit_score(y_val, preds)
                
        except Exception as e:
            print(f"GPU Error in trial: {e}")
            return -1000.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print(f"Best CatBoost params: {study.best_params}")
    return study.best_params

def optimize_mlp(X, y, n_trials=20, timeout=300):
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

def optimize_lstm(X_train, y_train, X_val, y_val, n_trials=15, timeout=600):
    """
    Optimizes LSTM hyperparameters.
    """
    print(f"--- Starting LSTM Optimization ({n_trials} trials) ---")
    
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
        preds = (preds_proba >= 0.5).astype(int)
        
        return calculate_profit_score(y_val, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print(f"Best LSTM params: {study.best_params}")
    return study.best_params
