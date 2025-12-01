
from sklearn.neural_network import MLPClassifier
import joblib
import os
import numpy as np

def train_model_mlp(X_train, y_p15_train, y_p3_train):
    """
    Trains MLP models.
    """
    # Filter features for MLP: Use ONLY Raw Lags and HMM State
    # This forces the Neural Network to learn its own representations without our "hand-crafted" features.
    # We want diversity in the ensemble.
    
    feature_cols = [col for col in X_train.columns if col.startswith('raw_lag_') or col == 'hmm_state']
    X_train_filtered = X_train[feature_cols].copy()
    
    print(f"MLP Input Features: {len(feature_cols)} (Raw Lags + HMM Only)")
    
    # Split
    split_idx = int(len(X_train_filtered) * 0.85)
    X_t, X_val = X_train_filtered.iloc[:split_idx], X_train_filtered.iloc[split_idx:]
    y_p15_t, y_p15_val = y_p15_train[:split_idx], y_p15_train[split_idx:]
    y_p3_t, y_p3_val = y_p3_train[:split_idx], y_p3_train[split_idx:]
    
    # P1.5 Model
    print("Training MLP (P1.5)...")
    clf_p15 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', 
                            solver='adam', max_iter=500, early_stopping=True, verbose=True)
    clf_p15.fit(X_t, y_p15_t)
    
    # P3.0 Model
    print("Training MLP (P3.0)...")
    clf_p3 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', 
                           solver='adam', max_iter=500, early_stopping=True, verbose=True)
    clf_p3.fit(X_t, y_p3_t)
    
    return clf_p15, clf_p3, feature_cols # Return feature cols to save/use during prediction

def save_mlp_models(model_p15, model_p3, feature_cols, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    joblib.dump(model_p15, os.path.join(output_dir, 'modelE_p15.pkl'))
    joblib.dump(model_p3, os.path.join(output_dir, 'modelE_p3.pkl'))
    joblib.dump(feature_cols, os.path.join(output_dir, 'modelE_features.pkl'))
    print(f"MLP models saved to {output_dir}")

```
