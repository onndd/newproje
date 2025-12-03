
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib

def create_sequences(values, seq_length=200):
    """
    Creates sequences for LSTM.
    """
    X = []
    y_p15 = []
    y_p3 = []
    y_val = []
    
    for i in range(seq_length, len(values)):
        seq = values[i-seq_length:i]
        X.append(seq)
        target = values[i]
        y_val.append(target)
        y_p15.append(1 if target >= 1.5 else 0)
        y_p3.append(1 if target >= 3.0 else 0)
        
    return np.array(X), np.array(y_p15), np.array(y_p3), np.array(y_val)

def build_lstm_model(seq_length):
    """
    Builds a simplified LSTM model.
    """
    from tensorflow.keras.layers import Input
    
    model = Sequential()
    model.add(Input(shape=(seq_length, 1)))
    # Simplified architecture for ~15k samples
    model.add(LSTM(128, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model_lstm(values, seq_length=200, epochs=20, batch_size=64):
    """
    Trains LSTM models for P1.5 and P3 with NO DATA LEAKAGE.
    """
    # 1. Split Data Chronologically FIRST
    # We need to reserve the last 15% for validation
    # FIX: Add a GAP of seq_length to ensure absolutely no overlap/leakage
    split_idx = int(len(values) * 0.85)
    
    train_values = values[:split_idx]
    # Gap: Skip seq_length samples to avoid any window overlap
    val_values = values[split_idx + seq_length:] 
    
    if len(val_values) < seq_length + 10:
        print("Warning: Validation set too small after gap. Reducing gap.")
        val_values = values[split_idx:] # Fallback
    
    # 2. Fit Scaler ONLY on Training Data
    # FIX: Apply Log Transformation first to handle outliers (e.g. 5000x)
    train_values_log = np.log1p(train_values)
    val_values_log = np.log1p(val_values)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_values_log.reshape(-1, 1))
    
    # Transform
    train_scaled = scaler.transform(train_values_log.reshape(-1, 1))
    val_scaled = scaler.transform(val_values_log.reshape(-1, 1))
    
    # 3. Create Sequences Separately
    # Train sequences: strictly from train data
    X_train, y_p15_train, y_p3_train, _ = create_sequences(train_scaled, seq_length)
    
    # Validation sequences:
    # Since we added a gap, we don't need context from train anymore.
    # We treat validation as a completely independent future segment.
    X_val, y_p15_val, y_p3_val, _ = create_sequences(val_scaled, seq_length)
    
    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    # Compute Class Weights
    from sklearn.utils.class_weight import compute_class_weight
    
    # P1.5 Weights
    classes_p15 = np.unique(y_p15_train)
    weights_p15 = compute_class_weight(class_weight='balanced', classes=classes_p15, y=y_p15_train)
    class_weight_p15 = dict(zip(classes_p15, weights_p15))
    print(f"P1.5 Class Weights: {class_weight_p15}")
    
    # P3.0 Weights
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
        
    model_p15 = load_model(p15_path)
    model_p3 = load_model(p3_path)
    scaler = joblib.load(scaler_path)
    
    return model_p15, model_p3, scaler
