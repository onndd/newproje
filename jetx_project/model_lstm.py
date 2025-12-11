
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib

def create_sequences(input_values, target_values, seq_length=200):
    """
    Creates sequences for LSTM.
    input_values: Scaled values for X
    target_values: Raw values for y (to check >= 1.5)
    """
    X = []
    y_p15 = []
    y_p3 = []
    y_val = []
    
    # Ensure lengths match (they should if passed correctly)
    min_len = min(len(input_values), len(target_values))
    
    for i in range(seq_length, min_len):
        seq = input_values[i-seq_length:i]
        X.append(seq)
        
        target = target_values[i] # Use RAW value for target
        y_val.append(target)
        y_p15.append(1 if target >= 1.5 else 0)
        y_p3.append(1 if target >= 3.0 else 0)
        
    return np.array(X), np.array(y_p15), np.array(y_p3), np.array(y_val)

def build_lstm_model(seq_length, units=128, dropout=0.2, dense_units=64):
    """
    Basit iki katmanlı LSTM mimarisi.
    Giriş: (seq_length, 1), Çıkış: tek olasılık başı.
    """
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(LSTM(int(units/2), return_sequences=False))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    return model

class BinaryFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name='binary_focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss = - alpha_t * tf.math.pow(1 - p_t, self.gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)

def train_model_lstm(values, seq_length=200, epochs=15, batch_size=128, params_p15=None, params_p3=None):
    """
    Trains LSTM models for P1.5 and P3 with NO DATA LEAKAGE.
    """
    # 1. Strict Chronological Split (Raw Data)
    n_total = len(values)
    train_end = int(n_total * 0.70)
    val_start = int(n_total * 0.75)
    val_end = int(n_total * 0.85)
    
    raw_train = values[:train_end]
    raw_val = values[val_start:val_end]
    
    print(f"LSTM Split: Train ({len(raw_train)}), Val ({len(raw_val)})")
    
    # 2. Fit Scaler ONLY on Training Data
    # Log Transform first
    train_log = np.log1p(raw_train)
    val_log = np.log1p(raw_val)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_log.reshape(-1, 1))
    
    train_scaled = scaler.transform(train_log.reshape(-1, 1))
    val_scaled = scaler.transform(val_log.reshape(-1, 1))
    
    # 3. Create Sequences Separately
    # Pass SCALED for X, and RAW for y
    X_train, y_p15_train, y_p3_train, _ = create_sequences(train_scaled, raw_train, seq_length)
    X_val, y_p15_val, y_p3_val, _ = create_sequences(val_scaled, raw_val, seq_length)
    
    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    print(f"LSTM Sequences: Train ({len(X_train)}), Val ({len(X_val)})")
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    
    # Define Metrics
    metrics = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    
    # --- P1.5 Model ---
    print("Training LSTM (P1.5)...")
    
    # Defaults
    p15_args = {
        'units': 128, 'dropout': 0.2, 'dense_units': 64, 
        'learning_rate': 0.001, 'batch_size': batch_size
    }
    if params_p15:
        # Clean params if needed
        clean_params = {k: v for k, v in params_p15.items() if k in ['units1', 'dropout1', 'units2', 'dropout2', 'dense_units', 'lr', 'batch_size']}
        # Map optuna names to function names if necessary or just update
        # Our optimization.py uses 'units1', 'units2' etc. but `build_lstm_model` uses simple `units`.
        # We need to adapt the build function or the args.
        # Let's adapt args to match what build_lstm_model expects (simple 2-layer) or update build_lstm_model to be flexible.
        # For simplicity, we'll map 'units1' -> 'units' if present.
        
        if 'units1' in params_p15: p15_args['units'] = params_p15['units1']
        if 'dropout1' in params_p15: p15_args['dropout'] = params_p15['dropout1']
        if 'dense_units' in params_p15: p15_args['dense_units'] = params_p15['dense_units']
        if 'lr' in params_p15: p15_args['learning_rate'] = params_p15['lr']
        if 'batch_size' in params_p15: p15_args['batch_size'] = params_p15['batch_size']
        
    model_p15 = build_lstm_model(seq_length, units=p15_args['units'], dropout=p15_args['dropout'], dense_units=p15_args['dense_units'])
    
    opt = tf.keras.optimizers.Adam(learning_rate=p15_args['learning_rate'])
    
    # Use Focal Loss for P1.5 to fix "Always No" issue
    # Alpha 0.6 means we give slightly more weight to Class 1 (if 1 is minority importance) or handle imbalance
    model_p15.compile(optimizer=opt, loss=BinaryFocalLoss(gamma=2.0, alpha=0.60), metrics=metrics)
    
    model_p15.fit(X_train, y_p15_train, validation_data=(X_val, y_p15_val),
                  epochs=epochs, batch_size=p15_args['batch_size'], callbacks=callbacks, verbose=1)
                  # Removed class_weight because Focal Loss handles it internally via Alpha
                  
    # --- P3.0 Model ---
    print("Training LSTM (P3.0)...")
    p3_args = {
        'units': 128, 'dropout': 0.2, 'dense_units': 64, 
        'learning_rate': 0.001, 'batch_size': batch_size
    }
    if params_p3:
        if 'units1' in params_p3: p3_args['units'] = params_p3['units1']
        if 'dropout1' in params_p3: p3_args['dropout'] = params_p3['dropout1']
        if 'dense_units' in params_p3: p3_args['dense_units'] = params_p3['dense_units']
        if 'lr' in params_p3: p3_args['learning_rate'] = params_p3['lr']
        if 'batch_size' in params_p3: p3_args['batch_size'] = params_p3['batch_size']
        
    model_p3 = build_lstm_model(seq_length, units=p3_args['units'], dropout=p3_args['dropout'], dense_units=p3_args['dense_units'])
    
    opt_3 = tf.keras.optimizers.Adam(learning_rate=p3_args['learning_rate'])
    # P3 is also imbalanced, Focal Loss helps
    model_p3.compile(optimizer=opt_3, loss=BinaryFocalLoss(gamma=2.0, alpha=0.70), metrics=metrics)
    
    model_p3.fit(X_train, y_p3_train, validation_data=(X_val, y_p3_val),
                 epochs=epochs, batch_size=p3_args['batch_size'], callbacks=callbacks, verbose=1)

                 
    # Detailed Reporting
    from sklearn.metrics import confusion_matrix, classification_report
    
    # P1.5 Report
    print("\n--- LSTM P1.5 Report ---")
    preds_p15_prob = model_p15.predict(X_val)
    preds_p15 = (preds_p15_prob > 0.5).astype(int)
    cm_p15 = confusion_matrix(y_p15_val, preds_p15)
    print(f"Confusion Matrix (P1.5):\n{cm_p15}")
    if cm_p15.shape == (2, 2):
        tn, fp, fn, tp = cm_p15.ravel()
        print(f"Correctly Predicted >1.5x: {tp}/{tp+fn} (Recall: {tp/(tp+fn):.2%})")
        print(f"False Alarms: {fp}/{tp+fp} (Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.2%})")
    print(classification_report(y_p15_val, preds_p15))

    # P3.0 Report
    print("\n--- LSTM P3.0 Report ---")
    preds_p3_prob = model_p3.predict(X_val)
    preds_p3 = (preds_p3_prob > 0.5).astype(int)
    cm_p3 = confusion_matrix(y_p3_val, preds_p3)
    print(f"Confusion Matrix (P3.0):\n{cm_p3}")
    if cm_p3.shape == (2, 2):
        tn, fp, fn, tp = cm_p3.ravel()
        print(f"Correctly Predicted >3.0x: {tp}/{tp+fn} (Recall: {tp/(tp+fn):.2%})")
        print(f"False Alarms: {fp}/{tp+fp} (Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.2%})")
    print(classification_report(y_p3_val, preds_p3))
                 
    return model_p15, model_p3, scaler

def save_lstm_models(model_p15, model_p3, scaler, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_p15.save(os.path.join(output_dir, 'modelC_p15.h5'))
    model_p3.save(os.path.join(output_dir, 'modelC_p3.h5'))
    joblib.dump(scaler, os.path.join(output_dir, 'modelC_scaler.pkl'))
    print(f"LSTM models saved to {output_dir}")

def load_lstm_models(model_dir='.'):
    from tensorflow.keras.models import load_model
    
    p15_path = os.path.join(model_dir, 'modelC_p15.h5')
    p3_path = os.path.join(model_dir, 'modelC_p3.h5')
    scaler_path = os.path.join(model_dir, 'modelC_scaler.pkl')
    
    if not os.path.exists(p15_path) or not os.path.exists(p3_path) or not os.path.exists(scaler_path):
        return None, None, None
        
    # Fix: compile=False for Apple Silicon/Inference safety
    # Add BinaryFocalLoss to custom_objects just in case, though compile=False usually skips it
    model_p15 = load_model(p15_path, custom_objects={'BinaryFocalLoss': BinaryFocalLoss}, compile=False)
    model_p3 = load_model(p3_path, custom_objects={'BinaryFocalLoss': BinaryFocalLoss}, compile=False)
    scaler = joblib.load(scaler_path)
    
    return model_p15, model_p3, scaler
