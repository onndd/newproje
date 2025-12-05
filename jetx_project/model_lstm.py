
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

# ... (build_lstm_model remains same) ...

def train_model_lstm(values, seq_length=200, epochs=5, batch_size=128):
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
    
    # Compute Class Weights (No Upsampling)
    from sklearn.utils.class_weight import compute_class_weight
    
    classes_p15 = np.unique(y_p15_train)
    weights_p15 = compute_class_weight(class_weight='balanced', classes=classes_p15, y=y_p15_train)
    class_weight_p15 = dict(zip(classes_p15, weights_p15))
    print(f"P1.5 Class Weights: {class_weight_p15}")
    
    classes_p3 = np.unique(y_p3_train)
    weights_p3 = compute_class_weight(class_weight='balanced', classes=classes_p3, y=y_p3_train)
    class_weight_p3 = dict(zip(classes_p3, weights_p3))
    print(f"P3.0 Class Weights: {class_weight_p3}")

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    
    # Define Metrics
    metrics = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    
    print("Training LSTM (P1.5)...")
    model_p15 = build_lstm_model(seq_length)
    # Re-compile to add new metrics
    model_p15.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    model_p15.fit(X_train, y_p15_train, validation_data=(X_val, y_p15_val),
                  epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1,
                  class_weight=class_weight_p15)
                  
    print("Training LSTM (P3.0)...")
    model_p3 = build_lstm_model(seq_length)
    # Re-compile to add new metrics
    model_p3.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    model_p3.fit(X_train, y_p3_train, validation_data=(X_val, y_p3_val),
                 epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1,
                 class_weight=class_weight_p3)
                 
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
    model_p15 = load_model(p15_path, compile=False)
    model_p3 = load_model(p3_path, compile=False)
    scaler = joblib.load(scaler_path)
    
    return model_p15, model_p3, scaler
