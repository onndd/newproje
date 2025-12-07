
from sklearn.neural_network import MLPClassifier
import joblib
import os
import numpy as np
import pandas as pd

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
    
    # Compute Sample Weights for Class Balancing
    from sklearn.utils.class_weight import compute_sample_weight
    
    # P1.5 Model
    # Manual Oversampling for Class Balancing (since MLPClassifier doesn't support class_weight)
    def balance_data(X, y, weight_1):
        # Separate classes
        mask_0 = y == 0
        mask_1 = y == 1
        X_0, y_0 = X[mask_0], y[mask_0]
        X_1, y_1 = X[mask_1], y[mask_1]
        
        # Determine number of copies for class 1
        # If weight_1 is 2.0, we want to double the presence of class 1 roughly
        # Or simpler: duplicate class 1 'weight_1' times (int)
        import math
        repeats = math.ceil(weight_1)
        
        X_1_balanced = pd.concat([X_1] * repeats, axis=0)
        y_1_balanced = np.tile(y_1, repeats) # numpy array tile
        
        # Combine
        X_balanced = pd.concat([X_0, X_1_balanced], axis=0)
        y_balanced = np.concatenate([y_0, y_1_balanced])
        
        # Shuffle
        perm = np.random.permutation(len(X_balanced))
        return X_balanced.iloc[perm], y_balanced[perm]
    
    # P1.5 Model (Weight 1.0 -> 2.0 for positives)
    print("Training MLP (P1.5)...")
    X_t_p15, y_p15_t_balanced = balance_data(X_t, y_p15_t, 2.0)
    
    clf_p15 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', 
                            solver='adam', alpha=0.01, learning_rate_init=0.001,
                            max_iter=500, early_stopping=True, verbose=True)
    clf_p15.fit(X_t_p15, y_p15_t_balanced)
    
    # P3.0 Model (Weight 1.0 -> 2.0 for positives)
    print("Training MLP (P3.0)...")
    X_t_p3, y_p3_t_balanced = balance_data(X_t, y_p3_t, 2.0)
    
    clf_p3 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', 
                           solver='adam', alpha=0.01, learning_rate_init=0.001,
                           max_iter=500, early_stopping=True, verbose=True)
    clf_p3.fit(X_t_p3, y_p3_t_balanced)
    
    # Detailed Reporting
    from sklearn.metrics import confusion_matrix, classification_report
    
    # P1.5 Report
    preds_p15 = clf_p15.predict(X_val)
    print("\n--- MLP P1.5 Report ---")
    cm_p15 = confusion_matrix(y_p15_val, preds_p15)
    print(f"Confusion Matrix (P1.5):\n{cm_p15}")
    if cm_p15.shape == (2, 2):
        tn, fp, fn, tp = cm_p15.ravel()
        print(f"Correctly Predicted >1.5x: {tp}/{tp+fn} (Recall: {tp/(tp+fn):.2%})")
        print(f"False Alarms: {fp}/{tp+fp} (Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.2%})")
    print(classification_report(y_p15_val, preds_p15))

    # P3.0 Report
    preds_p3 = clf_p3.predict(X_val)
    print("\n--- MLP P3.0 Report ---")
    cm_p3 = confusion_matrix(y_p3_val, preds_p3)
    print(f"Confusion Matrix (P3.0):\n{cm_p3}")
    if cm_p3.shape == (2, 2):
        tn, fp, fn, tp = cm_p3.ravel()
        print(f"Correctly Predicted >3.0x: {tp}/{tp+fn} (Recall: {tp/(tp+fn):.2%})")
        print(f"False Alarms: {fp}/{tp+fp} (Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.2%})")
    print(classification_report(y_p3_val, preds_p3))
    
    return clf_p15, clf_p3, feature_cols # Return feature cols to save/use during prediction

def save_mlp_models(model_p15, model_p3, feature_cols, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    joblib.dump(model_p15, os.path.join(output_dir, 'modelE_p15.pkl'))
    joblib.dump(model_p3, os.path.join(output_dir, 'modelE_p3.pkl'))
    joblib.dump(feature_cols, os.path.join(output_dir, 'modelE_cols.pkl'))
    print(f"MLP models saved to {output_dir}")

def load_mlp_models(model_dir='.'):
    p15_path = os.path.join(model_dir, 'modelE_p15.pkl')
    p3_path = os.path.join(model_dir, 'modelE_p3.pkl')
    cols_path = os.path.join(model_dir, 'modelE_cols.pkl')
    
    if not os.path.exists(p15_path) or not os.path.exists(p3_path) or not os.path.exists(cols_path):
        return None, None, None
        
    model_p15 = joblib.load(p15_path)
    model_p3 = joblib.load(p3_path)
    feature_cols = joblib.load(cols_path)
    
    return model_p15, model_p3, feature_cols
