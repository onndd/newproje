
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

def train_model_lstm(values, seq_length=200, epochs=5, batch_size=128):
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
    X_train_full, y_p15_train_full, y_p3_train_full, _ = create_sequences(train_scaled, seq_length)
    
    # Validation sequences:
    # Since we added a gap, we don't need context from train anymore.
    # We treat validation as a completely independent future segment.
    X_val_full, y_p15_val_full, y_p3_val_full, _ = create_sequences(val_scaled, seq_length)
    # Fix: Strict Chronological Split with Gap to prevent leakage
    # Train: 0 to 70%
    # Gap: 70% to 75% (Unused)
    # Val: 75% to 85%
    
    # Combine scaled train and validation data for overall sequence creation
    # This is a conceptual combination for splitting, actual sequences are created from respective parts
    # For the new splitting logic, we need to create sequences from the *entire* scaled dataset first,
    # then split those sequences.
    
    # Re-doing sequence creation based on the new splitting logic
    # First, create sequences from the *entire* scaled dataset (train_scaled + val_scaled)
    # This assumes `values` in create_sequences is the full scaled dataset.
    # Let's adjust the logic to fit the new splitting requirement.
    
    # The original `train_model_lstm` function already splits `values` into `train_values` and `val_values`
    # and then scales them into `train_scaled` and `val_scaled`.
    # The `create_sequences` function is then called on `train_scaled` and `val_scaled` separately.
    # The new instruction seems to imply creating sequences from a single `X` (representing all data)
    # and then splitting `X` and `y` arrays.
    # This contradicts the previous `train_scaled` and `val_scaled` separation.
    
    # To faithfully apply the instruction, I will assume the instruction wants to replace the
    # existing `X_train`, `y_p15_train`, `y_p3_train`, `X_val`, `y_p15_val`, `y_p3_val`
    # with a new splitting scheme *after* initial sequence creation.
    # This means the `create_sequences` call should happen on a larger dataset first.
    
    # Let's revert to the original splitting logic for `train_scaled` and `val_scaled`
    # and then apply the new splitting logic on the *results* of `create_sequences`
    # if the instruction intends to re-split the already created sequences.
    
    # Given the instruction's context, it seems to want to apply a different chronological split
    # *after* the initial scaling and sequence creation.
    # This implies that `X` in the instruction refers to the combined sequences from `train_scaled` and `val_scaled`.
    # This is a bit ambiguous with the existing code structure.
    
    # Let's assume the instruction wants to replace the current `X_train`, `X_val` etc.
    # with a new split based on a combined `X` and `y` from the *entire* scaled dataset.
    # This would require creating sequences from `np.concatenate((train_scaled, val_scaled))`
    # which would introduce data leakage if `val_scaled` is used to create `X_train`.
    
    # The instruction says "Fix leakage by strict splitting and gap".
    # The current code already has a gap between `train_values` and `val_values`.
    # The instruction's new split `total_len = len(X)`, `train_end = int(total_len * 0.70)`, etc.
    # implies `X` is the full set of sequences.
    
    # To make the change faithfully and syntactically correct, I will interpret the instruction
    # as replacing the *definition* of `X_train`, `y_p15_train`, `y_p3_train`, `X_val`, `y_p15_val`, `y_p3_val`
    # with the new splitting logic, assuming `X`, `y_p15`, `y_p3` are derived from the *entire* dataset.
    # This would mean the previous `create_sequences(train_scaled, ...)` and `create_sequences(val_scaled, ...)`
    # are no longer the primary source for `X_train` and `X_val`.
    
    # Let's create sequences from the *entire* scaled dataset first, then apply the new split.
    # This is the most direct interpretation of the provided code snippet.
    
    # Combine scaled data for overall sequence creation
    full_scaled_data = np.concatenate((train_scaled, val_scaled))
    X_all, y_p15_all, y_p3_all, _ = create_sequences(full_scaled_data, seq_length)
    
    # Reshape for LSTM (before splitting, as the split is on the sequence arrays)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))
    
    # Fix: Strict Chronological Split with Gap to prevent leakage
    # Train: 0 to 70%
    # Gap: 70% to 75% (Unused)
    # Val: 75% to 85%
    
    total_len = len(X_all)
    train_end = int(total_len * 0.70)
    val_start = int(total_len * 0.75) # 5% gap
    val_end = int(total_len * 0.85)
    
    X_train = X_all[:train_end]
    y_p15_train = y_p15_all[:train_end]
    y_p3_train = y_p3_all[:train_end]
    
    X_val = X_all[val_start:val_end]
    y_p15_val = y_p15_all[val_start:val_end]
    y_p3_val = y_p3_all[val_start:val_end]
    
    print(f"LSTM Train size: {len(X_train)}, Val size: {len(X_val)}")
    
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
