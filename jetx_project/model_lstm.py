
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
    # 1. Split Data FIRST (Train/Val)
    # We need to reserve the last 15% for validation
    split_idx = int(len(values) * 0.85)
    
    train_values = values[:split_idx]
    val_values = values[split_idx:]
    
    # 2. Fit Scaler ONLY on Training Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit on train
    scaler.fit(train_values.reshape(-1, 1))
    
    # Transform both
    # Note: We need to handle the sequence generation carefully.
    # For training sequences, we use train_values.
    # For validation sequences, we need some overlap from train to start the first val sequence.
    
    # Let's scale the entire array now using the scaler fitted on TRAIN
    values_scaled = scaler.transform(values.reshape(-1, 1))
    
    # 3. Create Sequences
    # We create sequences from the FULL scaled array, then split again based on index
    # This ensures continuity but we must be careful not to leak future info into scaler (which we avoided above)
    
    X = []
    y_p15 = []
    y_p3 = []
    
    # Targets are derived from raw values (to avoid scaling artifacts in classification logic)
    # But we need to align indices.
    
    for i in range(seq_length, len(values)):
        seq = values_scaled[i-seq_length:i]
        X.append(seq)
        
        target = values[i]
        y_p15.append(1 if target >= 1.5 else 0)
        y_p3.append(1 if target >= 3.0 else 0)
        
    X = np.array(X)
    y_p15 = np.array(y_p15)
    y_p3 = np.array(y_p3)
    
    # 4. Split Sequences
    # The split_idx for values corresponds to a specific index in X
    # values index 'i' corresponds to X index 'i - seq_length'
    # We want X_train to include sequences where the TARGET is in train_values
    
    # train_values ends at split_idx. So last target index is split_idx - 1.
    # X index = (split_idx - 1) - seq_length
    
    # Let's just split X based on the ratio again, it's safer and easier
    split_seq_idx = int(len(X) * 0.85)
    
    X_train, X_val = X[:split_seq_idx], X[split_seq_idx:]
    y_p15_train, y_p15_val = y_p15[:split_seq_idx], y_p15[split_seq_idx:]
    y_p3_train, y_p3_val = y_p3[:split_seq_idx], y_p3[split_seq_idx:]
    
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
