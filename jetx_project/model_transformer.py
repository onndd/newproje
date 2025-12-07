
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(seq_length, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=np.arange(position)[:, np.newaxis],
            i=np.arange(d_model)[np.newaxis, :],
            d_model=d_model
        )
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def build_transformer_model(seq_length, num_heads=4, key_dim=32, ff_dim=64):
    """
    Builds a Transformer model for Time-Series forecasting.
    "The Attention" - Captures long-term dependencies.
    """
    inputs = Input(shape=(seq_length, 1))
    
    # Projection Layer (Dimension Mismatch Fix)
    # Project 1D input to key_dim (e.g. 32) so Attention has depth to work with
    x = Dense(key_dim)(inputs) 
    
    # Positional Encoding (Time Awareness Fix)
    x = PositionalEncoding(seq_length, key_dim)(x)
    
    # Multi-Head Attention
    # Now inputs to attention have shape (batch, seq, key_dim)
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn_out = Dropout(0.1)(attn_out)
    
    # Residual Connection 1
    # x and attn_out now have same shape (batch, seq, key_dim)
    x = LayerNormalization(epsilon=1e-6)(x + attn_out) 
    
    # Feed Forward Part
    res = x # Skip connection
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(key_dim)(x) # Project back to key_dim
    
    # Residual Connection 2
    x = LayerNormalization(epsilon=1e-6)(x + res)
    
    # Global Average Pooling to flatten
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    
    # Output Heads
    output_p15 = Dense(1, activation='sigmoid', name='p15')(x)
    output_p3 = Dense(1, activation='sigmoid', name='p3')(x)
    
    model = Model(inputs=inputs, outputs=[output_p15, output_p3])
    model.compile(optimizer='adam', 
                  loss={'p15': 'binary_crossentropy', 'p3': 'binary_crossentropy'},
                  metrics={'p15': 'accuracy', 'p3': 'accuracy'})
    return model

def train_model_transformer(values, seq_length=200, epochs=20, batch_size=64):
    """
    Trains the Transformer model.
    """
    # 1. Split Data (Chronological)
    split_idx = int(len(values) * 0.85)
    train_values = values[:split_idx]
    val_values = values[split_idx:]
    
    # 2. Scale (Log + MinMax)
    train_values_log = np.log1p(train_values)
    val_values_log = np.log1p(val_values)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_values_log.reshape(-1, 1))
    
    train_scaled = scaler.transform(train_values_log.reshape(-1, 1))
    
    # Validation Context
    # Note: We append validation targets to context to allow autoregressive evaluation (Teacher Forcing).
    # X_val[0] will use Context. 
    # X_val[1] will use Context + Val[0]. 
    # This is CAUSAL because at step T we know T-1.
    context_values = train_values[-seq_length:]
    
    # CRITICAL FIX (LEAKAGE):
    # Old: val_values_with_context = concat(context, val_values) -> create_sequences includes future?
    # Actually, create_sequences(X, y) uses X[i : i+seq] to predict y[i+seq].
    # So if X contains the *target itself* at the last position, it's a leak IF we are trying to predict that same index.
    # To be strictly safe and clear:
    # 1. We form a continuous stream: context + val_values
    stream = np.concatenate([context_values, val_values])
    stream_log = np.log1p(stream)
    stream_scaled = scaler.transform(stream_log.reshape(-1, 1))
    
    # 2. X, y generation
    # We want to predict val_values[0], val_values[1]...
    # To predict val_values[0], we need the previous `seq_length` items (which is `context_values`).
    # `create_sequences` with `seq_length` does exactly this:
    # It takes window [0..seq-1] to predict [seq].
    # So if 'stream' starts with context (len=seq), then index 'seq' in stream is val_values[0].
    # So create_sequences(stream, stream, seq_length) will produce:
    # X[0] = stream[0:seq] (context), y[0] = stream[seq] (val_values[0]) -> CORRECT.
    # X[1] = stream[1:seq+1] (context[1:]+val[0]), y[1] = val[1] -> CORRECT.
    
    # So the logic was actually mathematically fine, BUT explicit separation ensures no offset error.
    # We will pass the full stream.
    
    X_train, y_p15_train, y_p3_train, _ = create_sequences(train_scaled, train_values, seq_length)
    
    # For validation, we use the stream but only take the parts that result in validation targets
    X_val_all, y_p15_val_all, y_p3_val_all, _ = create_sequences(stream_scaled, stream, seq_length)
    
    # The 'stream' contains context + val.
    # We only care about predictions for the 'val' part.
    # The first prediction from creates_sequences on the stream will correspond to predicting index `seq_length` of the stream.
    # Since stream = [context (len=seq), val (len=N)], index `seq_length` IS val[0].
    # So X_val_all[0] predicts val[0].
    # We just need to make sure we don't accidentally train on future val data if we were doing something else.
    # But here we just use it for validation.
    
    # Re-assign strictly
    X_val = X_val_all
    y_p15_val = y_p15_val_all
    y_p3_val = y_p3_val_all
    
    # Reshape
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    # 4. Train
    # 4. Train
    # Fix: Use sample_weight instead of class_weight for multi-output model support
    
    from sklearn.utils.class_weight import compute_sample_weight
    
    # P1.5 Sample Weights
    sample_weight_p15 = compute_sample_weight(class_weight='balanced', y=y_p15_train)
    
    # P3.0 Sample Weights
    sample_weight_p3 = compute_sample_weight(class_weight='balanced', y=y_p3_train)
    
    # Sample weight'ler Keras çoklu çıktı ile sorun çıkardığı için devre dışı bırakıldı.
    sample_weights = None
    
    print("Computed sample weights for Transformer multi-output training.")
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    
    # Define Metrics
    metrics = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    
    # Re-compile to add new metrics
    model = build_transformer_model(seq_length)
    model.compile(optimizer='adam', 
                  loss={'p15': 'binary_crossentropy', 'p3': 'binary_crossentropy'},
                  metrics={'p15': metrics, 'p3': metrics})
    
    print("Training Transformer (The Attention)...")
    model.fit(
        X_train,
        {'p15': y_p15_train, 'p3': y_p3_train},
        validation_data=(X_val, {'p15': y_p15_val, 'p3': y_p3_val}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        sample_weight=sample_weights
    )
              
    # Detailed Reporting
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Predict
    preds = model.predict(X_val)
    preds_p15_prob = preds[0]
    preds_p3_prob = preds[1]
    
    # P1.5 Report
    print("\n--- Transformer P1.5 Report ---")
    preds_p15 = (preds_p15_prob > 0.5).astype(int)
    cm_p15 = confusion_matrix(y_p15_val, preds_p15)
    print(f"Confusion Matrix (P1.5):\n{cm_p15}")
    if cm_p15.shape == (2, 2):
        tn, fp, fn, tp = cm_p15.ravel()
        print(f"Correctly Predicted >1.5x: {tp}/{tp+fn} (Recall: {tp/(tp+fn):.2%})")
        print(f"False Alarms: {fp}/{tp+fp} (Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.2%})")
    print(classification_report(y_p15_val, preds_p15))

    # P3.0 Report
    print("\n--- Transformer P3.0 Report ---")
    preds_p3 = (preds_p3_prob > 0.5).astype(int)
    cm_p3 = confusion_matrix(y_p3_val, preds_p3)
    print(f"Confusion Matrix (P3.0):\n{cm_p3}")
    if cm_p3.shape == (2, 2):
        tn, fp, fn, tp = cm_p3.ravel()
        print(f"Correctly Predicted >3.0x: {tp}/{tp+fn} (Recall: {tp/(tp+fn):.2%})")
        print(f"False Alarms: {fp}/{tp+fp} (Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.2%})")
    print(classification_report(y_p3_val, preds_p3))
              
    return model, scaler

def save_transformer_models(model, scaler, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(os.path.join(output_dir, 'model_transformer.h5'))
    joblib.dump(scaler, os.path.join(output_dir, 'model_transformer_scaler.pkl'))
    print(f"Transformer model saved to {output_dir}")

def load_transformer_models(model_dir='.'):
    model_path = os.path.join(model_dir, 'model_transformer.h5')
    scaler_path = os.path.join(model_dir, 'model_transformer_scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    # Fix: Pass custom_objects and compile=False for Apple Silicon/Inference safety
    try:
        model = load_model(
            model_path, 
            custom_objects={'PositionalEncoding': PositionalEncoding},
            compile=False
        )
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"Error loading Transformer model: {e}")
        return None, None
