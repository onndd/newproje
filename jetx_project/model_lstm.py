
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
    split_idx = int(len(values) * 0.85)
    
    train_values = values[:split_idx]
    val_values = values[split_idx:]
    
    # 2. Fit Scaler ONLY on Training Data
    # FIX: Apply Log Transformation first to handle outliers (e.g. 5000x)
    # This prevents the 1.0-2.0 range from being crushed.
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
    
    # DEBUG: Check for leakage
    print(f"DEBUG: X_train shape: {X_train.shape}")
    print(f"DEBUG: First sequence (last 5 values): {X_train[0, -5:, 0]}")
    print(f"DEBUG: First target (P1.5): {y_p15_train[0]}")
    print(f"DEBUG: First target (Value): {train_values[seq_length]}") # Corresponds to first target
    
    # Sanity Check: If X_train[0, -1] == target, we have leakage (assuming scaled)
    # But target is unscaled in print, X is scaled.
    # Let's check if X_train[i, -1] is exactly equal to X_train[i+1, -2] (overlap property)
    # This is expected.
    
    # Real leakage check: Does X[i] contain y[i]?
    # No easy way to check without inverse transform, but the print will help.
    
    # Validation sequences:
    # FIX: To avoid losing the first 'seq_length' samples of validation (Context Loss),
    # we need to prepend the last 'seq_length' samples of training data to validation data.
    # This provides the necessary history for the first validation prediction.
    
    # Get context from raw values (to be safe with scaling)
    context_values = train_values[-seq_length:]
    val_values_with_context = np.concatenate([context_values, val_values])
    
    # Scale the combined validation data using the SAME scaler fitted on train
    # FIX: Apply Log Transform first
    val_values_with_context_log = np.log1p(val_values_with_context)
    val_scaled_with_context = scaler.transform(val_values_with_context_log.reshape(-1, 1))
    
    # Create sequences
    # Now X_val will start predicting for the first element of original val_values
    X_val, y_p15_val, y_p3_val, _ = create_sequences(val_scaled_with_context, seq_length)
    
    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    
    print("Training LSTM (P1.5)...")
    model_p15 = build_lstm_model(seq_length)
    model_p15.fit(X_train, y_p15_train, validation_data=(X_val, y_p15_val),
                  epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
                  
    print("Training LSTM (P3.0)...")
    model_p3 = build_lstm_model(seq_length)
    model_p3.fit(X_train, y_p3_train, validation_data=(X_val, y_p3_val),
                 epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
                 
    return model_p15, model_p3, scaler

def save_lstm_models(model_p15, model_p3, scaler, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_p15.save(os.path.join(output_dir, 'modelC_p15.h5'))
    model_p3.save(os.path.join(output_dir, 'modelC_p3.h5'))
    joblib.dump(scaler, os.path.join(output_dir, 'modelC_scaler.pkl'))
    print(f"LSTM models saved to {output_dir}")
