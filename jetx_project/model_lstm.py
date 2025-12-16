
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

def train_model_lstm(values, params_p15=None, params_p3=None, scoring_params_p15=None, scoring_params_p3=None):
    """
    Trains LSTM models for P1.5 and P3 with NO DATA LEAKAGE.
    """
    
    # Define Helper for Threshold Search
    from sklearn.metrics import confusion_matrix, classification_report
    from .config import PROFIT_SCORING_WEIGHTS, SCORING_LSTM
    def find_best_threshold(y_true, y_prob, model_name, verbose=True, scoring_params=None):
        """
        Finds the optimal threshold based on Profit Scoring.
        """
        # Use provided scoring_params or default to SCORING_LSTM
        current_scoring_params = scoring_params if scoring_params is not None else SCORING_LSTM
            
        best_thresh = 0.5
        best_score = -float('inf')
        thresholds = np.arange(0.50, 0.99, 0.01)
        
        if verbose:
            print(f"\nScanning Thresholds for {model_name}...")
            
        for thresh in thresholds:
            preds = (y_prob > thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            score = (tp * current_scoring_params['TP']) - \
                    (fp * current_scoring_params['FP']) + \
                    (tn * current_scoring_params['TN']) - \
                    (fn * current_scoring_params['FN'])
            
            if score > best_score:
                best_score = score
                best_thresh = thresh
        
        if verbose:
            print(f"Best Threshold for {model_name}: {best_thresh:.2f} (Score: {best_score})")
        return best_thresh, best_score

    # Default values for seq_length, epochs, batch_size if not in params
    seq_length = params_p15.get('seq_length', 200) if params_p15 else 200
    epochs = params_p15.get('epochs', 15) if params_p15 else 15
    batch_size = params_p15.get('batch_size', 128) if params_p15 else 128

    # 1. Sequence Generation (From RAW data to preserve targets)
    X, y_p15_all, y_p3_all = create_rolling_window_sequences(values, seq_length)
    
    # 2. Split Size Calculation
    # X has fewer samples than 'values' due to seq_length padding
    n_samples = len(X)
    final_split = int(n_samples * 0.85)
    
    # 3. ROBUST SCALING (Fit on Train, Apply to All)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    
    # Reshape X for scaling: (Samples, Seq_Len, Features) -> (Samples*Seq_Len, Features)
    X_train_raw = X[:final_split]
    X_train_flat = X_train_raw.reshape(-1, 1)
    
    print("Fitting RobustScaler on LSTM Training Data...")
    scaler.fit(X_train_flat)
    
    # Transform full dataset
    X_flat = X.reshape(-1, 1)
    X_scaled_flat = scaler.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(X.shape)
    
    print(f"LSTM Data Scaled. Mean: {np.mean(X_scaled):.2f}, Std: {np.std(X_scaled):.2f}")
    
    # 4. Prepare Final Train/Val Sets
    X_train = X_scaled[:final_split]
    y_p15_train = y_p15_all[:final_split]
    y_p3_train = y_p3_all[:final_split]
    
    X_val = X_scaled[final_split:]
    y_p15_val = y_p15_all[final_split:]
    y_p3_val = y_p3_all[final_split:]

    # --- Manual Class Weight Calculation (Surgical Fix) ---
    from sklearn.utils.class_weight import compute_class_weight
    
    # Calculate for P1.5 (Threshold 1.5)
    y_p15_all = (values >= 1.5).astype(int)
    # Calculate weights based on TRAIN portion only to avoid leakage
    y_p15_train_calc = y_p15_all[:final_split]
    classes_p15 = np.unique(y_p15_train_calc)
    weights_p15 = compute_class_weight(class_weight='balanced', classes=classes_p15, y=y_p15_train_calc)
    class_weights_p15 = dict(zip(classes_p15, weights_p15))
    
    # INFLATE
    if 1 in class_weights_p15:
        class_weights_p15[1] = class_weights_p15[1] * 2.0
        print(f"LSTM P1.5 Surgical Fix: Inflated Class 1 weight to {class_weights_p15[1]:.4f}")

    # Calculate for P3.0 (Threshold 3.0)
    y_p3_all = (values >= 3.0).astype(int)
    y_p3_train_calc = y_p3_all[:final_split]
    classes_p3 = np.unique(y_p3_train_calc)
    weights_p3 = compute_class_weight(class_weight='balanced', classes=classes_p3, y=y_p3_train_calc)
    class_weights_p3 = dict(zip(classes_p3, weights_p3))
    
    # INFLATE
    if 1 in class_weights_p3:
        class_weights_p3[1] = class_weights_p3[1] * 2.0
        print(f"LSTM P3.0 Surgical Fix: Inflated Class 1 weight to {class_weights_p3[1]:.4f}")
    
    
    # --- ROLLING WINDOW CV ---
    print("\n[CV] Running 3-Fold Rolling Window CV for LSTM (Strict Scaler Isolation)...")
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores_p15 = []
    cv_scores_p3 = []
    
    # We must use values directly
    for fold, (train_idx, val_idx) in enumerate(tscv.split(values)):
        # Data Slicing
        raw_train = values[train_idx]
        
        # Validation needs context (past sequences)
        val_start = val_idx[0]
        context_start = max(0, val_start - seq_length)
        raw_val_expanded = values[context_start : val_idx[-1] + 1]
        
        # SCALER ISOLATION: Fit only on Train
        scaler_cv = MinMaxScaler(feature_range=(0, 1))
        train_log = np.log1p(raw_train)
        val_log_expanded = np.log1p(raw_val_expanded)
        
        scaler_cv.fit(train_log.reshape(-1, 1))
        
        train_scaled = scaler_cv.transform(train_log.reshape(-1, 1))
        val_scaled_expanded = scaler_cv.transform(val_log_expanded.reshape(-1, 1))
        
        # Sequence Generation
        X_t_cv, y_p15_t_cv, y_p3_t_cv, _ = create_sequences(train_scaled, raw_train, seq_length)
        X_v_cv, y_p15_v_cv, y_p3_v_cv, _ = create_sequences(val_scaled_expanded, raw_val_expanded, seq_length)
        
        if len(X_t_cv) < batch_size or len(X_v_cv) == 0:
            print(f"  Fold {fold+1}: Not enough data. Skipping.")
            continue
            
        X_t_cv = X_t_cv.reshape((X_t_cv.shape[0], X_t_cv.shape[1], 1))
        X_v_cv = X_v_cv.reshape((X_v_cv.shape[0], X_v_cv.shape[1], 1))
        
        # Setup Model (P1.5 Training)
        model_cv = build_lstm_model(seq_length)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model_cv.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        
        model_cv.fit(X_t_cv, y_p15_t_cv, validation_data=(X_v_cv, y_p15_v_cv), 
                     epochs=5, batch_size=batch_size, verbose=0, # Fast CV
                     callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
        
        probs = model_cv.predict(X_v_cv, verbose=0)
        _, score_p15 = find_best_threshold(y_p15_v_cv, probs, f"P1.5 Fold {fold+1}", verbose=False, scoring_params=scoring_params_p15)
        cv_scores_p15.append(score_p15)
        
        # P3.0 Training (Optional or separate loop? Do both)
        # Re-use X, just change Y
        model_cv_p3 = build_lstm_model(seq_length)
        opt_p3 = tf.keras.optimizers.Adam(learning_rate=0.001) # New optimizer instance
        model_cv_p3.compile(optimizer=opt_p3, loss='binary_crossentropy', metrics=['accuracy'])
        model_cv_p3.fit(X_t_cv, y_p3_t_cv, validation_data=(X_v_cv, y_p3_v_cv),
                       epochs=5, batch_size=batch_size, verbose=0,
                       callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
        probs_p3 = model_cv_p3.predict(X_v_cv, verbose=0)
        _, score_p3 = find_best_threshold(y_p3_v_cv, probs_p3, f"P3.0 Fold {fold+1}", verbose=False, scoring_params=scoring_params_p3)
        cv_scores_p3.append(score_p3)
        
        print(f"  Fold {fold+1} Scores -> P1.5: {score_p15:.2f}, P3.0: {score_p3:.2f}")

    print(f"[CV] Avg P1.5: {np.mean(cv_scores_p15):.2f}, Avg P3.0: {np.mean(cv_scores_p3):.2f}")
    # --------------------------------

    # Final Training (Full Data - 85/15)
    train_end = int(n_total * 0.85)
    raw_train = values[:train_end]
    raw_val = values[train_end:] # Note: We need context for val!
    
    print(f"Final LSTM Split: Train ({len(raw_train)}), Val ({len(raw_val)})")
    
    # Needs expanded val for creating X_val
    val_context = values[train_end - seq_length : ] 
    # Use context so X_val[0] matches target values[train_end]
    
    # 2. Fit Scaler ONLY on Training Data
    # Log Transform first
    train_log = np.log1p(raw_train)
    val_log_expanded = np.log1p(val_context)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_log.reshape(-1, 1))
    
    train_scaled = scaler.transform(train_log.reshape(-1, 1))
    val_scaled = scaler.transform(val_log_expanded.reshape(-1, 1))
    
    # 3. Create Sequences Separately
    # Pass SCALED for X, and RAW for y
    X_train, y_p15_train, y_p3_train, _ = create_sequences(train_scaled, raw_train, seq_length)
    X_val, y_p15_val, y_p3_val, _ = create_sequences(val_scaled, val_context, seq_length)
    
    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    print(f"LSTM Training Shapes: X_train={X_train.shape}, X_val={X_val.shape}")
    
    
    # Model P1.5
    print("\n--- Training LSTM (P1.5) ---")
    
    # Define Metrics
    metrics = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    
    # Defaults
    p15_args = {
        'units': 128, 'dropout': 0.2, 'dense_units': 64, 
        'learning_rate': 0.001, 'batch_size': batch_size, 'epochs': epochs
    }
    
    if params_p15:
        if 'units1' in params_p15: p15_args['units'] = params_p15['units1']
        if 'dropout1' in params_p15: p15_args['dropout'] = params_p15['dropout1']
        if 'dense_units' in params_p15: p15_args['dense_units'] = params_p15['dense_units']
        if 'lr' in params_p15: p15_args['learning_rate'] = params_p15['lr']
        if 'batch_size' in params_p15: p15_args['batch_size'] = params_p15['batch_size']
        if 'epochs' in params_p15: p15_args['epochs'] = params_p15['epochs']
        
    model_p15 = build_lstm_model(seq_length, units=p15_args['units'], dropout=p15_args['dropout'], dense_units=p15_args['dense_units'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=p15_args['learning_rate'])
    model_p15.compile(optimizer=optimizer, loss=BinaryFocalLoss(gamma=2.0, alpha=0.25), metrics=metrics)
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)]
    model_p15.fit(X_train, y_p15_train, validation_data=(X_val, y_p15_val), 
                  epochs=p15_args['epochs'], batch_size=p15_args['batch_size'], verbose=1, 
                  class_weight=class_weights_p15, callbacks=callbacks)

    # Detailed Reporting P1.5
    from sklearn.metrics import confusion_matrix, classification_report
    preds_p15_prob = model_p15.predict(X_val)
    # Use helper defined at top
    best_thresh_p15, best_score_p15 = find_best_threshold(y_p15_val, preds_p15_prob, "LSTM P1.5", scoring_params=scoring_params_p15)
    
    preds_p15 = (preds_p15_prob >= best_thresh_p15).astype(int)
    print(f"Confusion Matrix (P1.5):\n{confusion_matrix(y_p15_val, preds_p15)}")
    print(classification_report(y_p15_val, preds_p15))

    # Model P3.0
    print("\n--- Training LSTM (P3.0) ---")
    
    p3_args = {
        'units': 128, 'dropout': 0.2, 'dense_units': 64, 
        'learning_rate': 0.001, 'batch_size': batch_size, 'epochs': epochs
    }
    
    if params_p3:
        if 'units1' in params_p3: p3_args['units'] = params_p3['units1']
        if 'dropout1' in params_p3: p3_args['dropout'] = params_p3['dropout1']
        if 'dense_units' in params_p3: p3_args['dense_units'] = params_p3['dense_units']
        if 'lr' in params_p3: p3_args['learning_rate'] = params_p3['lr']
        if 'batch_size' in params_p3: p3_args['batch_size'] = params_p3['batch_size']
        if 'epochs' in params_p3: p3_args['epochs'] = params_p3['epochs']
        
    model_p3 = build_lstm_model(seq_length, units=p3_args['units'], dropout=p3_args['dropout'], dense_units=p3_args['dense_units'])
    opt_3 = tf.keras.optimizers.Adam(learning_rate=p3_args['learning_rate'])
    model_p3.compile(optimizer=opt_3, loss=BinaryFocalLoss(gamma=2.0, alpha=0.25), metrics=metrics)
    
    model_p3.fit(X_train, y_p3_train, validation_data=(X_val, y_p3_val), 
                  epochs=p3_args['epochs'], batch_size=p3_args['batch_size'], verbose=1, 
                  class_weight=class_weights_p3, callbacks=callbacks)

    # Detailed Reporting P3.0
    preds_p3_prob = model_p3.predict(X_val)
    best_thresh_p3, best_score_p3 = find_best_threshold(y_p3_val, preds_p3_prob, "LSTM P3.0", scoring_params=scoring_params_p3)
    
    preds_p3 = (preds_p3_prob >= best_thresh_p3).astype(int)
    print(f"Confusion Matrix (P3.0):\n{confusion_matrix(y_p3_val, preds_p3)}")
    print(classification_report(y_p3_val, preds_p3))
    
    cm_p3 = confusion_matrix(y_p3_val, preds_p3)
    print(f"Confusion Matrix (P3.0 @ {best_thresh_p3:.2f}):\n{cm_p3}")
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
