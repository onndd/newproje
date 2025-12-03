
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib

def build_transformer_model(seq_length, num_heads=4, key_dim=32, ff_dim=64):
    """
    Builds a Transformer model for Time-Series forecasting.
    "The Attention" - Captures long-term dependencies.
    """
    inputs = Input(shape=(seq_length, 1))
    
    # Multi-Head Attention
    x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    x = Dropout(0.1)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs) # Residual connection
    
    # Feed Forward Part
    res = x + inputs # Skip connection
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1)(x) # Project back to feature dim
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
                  metrics=['accuracy'])
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
    context_values = train_values[-seq_length:]
    val_values_with_context = np.concatenate([context_values, val_values])
    val_values_with_context_log = np.log1p(val_values_with_context)
    val_scaled_with_context = scaler.transform(val_values_with_context_log.reshape(-1, 1))
    
    # 3. Create Sequences
    from .model_lstm import create_sequences
    X_train, y_p15_train, y_p3_train, _ = create_sequences(train_scaled, seq_length)
    X_val, y_p15_val, y_p3_val, _ = create_sequences(val_scaled_with_context, seq_length)
    
    # Reshape
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    # 4. Train
    model = build_transformer_model(seq_length)
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    
    print("Training Transformer (The Attention)...")
    model.fit(X_train, {'p15': y_p15_train, 'p3': y_p3_train}, 
              validation_data=(X_val, {'p15': y_p15_val, 'p3': y_p3_val}),
              epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
              
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
        
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler
