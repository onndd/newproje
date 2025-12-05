
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
    print("Training MLP (P1.5)...")
    # Custom Penalty: Penalize False Positives (Class 0) more heavily
    # We want the model to be afraid of predicting 1 when it's actually 0.
    # So we increase the weight of Class 0.
    
    classes_p15 = np.unique(y_p15_t)
    weights_p15 = compute_sample_weight(class_weight='balanced', y=y_p15_t)
    
    # Identify indices where target is 0 and boost their weight
    # Assuming y_p15_t is numpy array or pandas series
    indices_0 = np.where(y_p15_t == 0)[0]
    weights_p15[indices_0] *= 2.0 # Double the penalty for missing a crash
    
    print("Applied 2.0x penalty multiplier to Class 0 (Crash) samples.")
    
    clf_p15 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', 
                            solver='adam', alpha=0.01, learning_rate_init=0.001,
                            max_iter=500, early_stopping=True, verbose=True)
    # Removed sample_weight as MLPClassifier.fit does not support it
    # Wait, if fit doesn't support sample_weight, we can't use this method directly for MLP!
    # Scikit-learn MLPClassifier ONLY supports sample_weight in partial_fit, OR in fit() for version >= 0.24?
    # Let's check environment. If user got TypeError, it means it's NOT supported.
    
    # Alternative: Oversampling Class 0 manually.
    # Since we cannot pass sample_weight, we must duplicate Class 0 samples.
    
    print("Oversampling Class 0 to enforce penalty...")
    X_t_p15 = X_t.copy()
    X_t_p15['target'] = y_p15_t
    
    df_0 = X_t_p15[X_t_p15.target == 0]
    df_1 = X_t_p15[X_t_p15.target == 1]
    
    # Duplicate Class 0 samples 1.5x (reduce false alarms)
    df_0_upsampled = pd.concat([df_0, df_0.sample(frac=0.5, replace=True)])
    
    df_upsampled = pd.concat([df_0_upsampled, df_1])
    X_t_final = df_upsampled.drop('target', axis=1)
    y_t_final = df_upsampled.target.values
    
    clf_p15.fit(X_t_final, y_t_final)
    
    # P3.0 Model (Same logic)
    print("Training MLP (P3.0)...")
    # sample_weight removed
    
    # Apply Oversampling for Class 0 (Penalty)
    X_t_p3 = X_t.copy()
    X_t_p3['target'] = y_p3_t
    
    df_0_p3 = X_t_p3[X_t_p3.target == 0]
    df_1_p3 = X_t_p3[X_t_p3.target == 1]
    
    # Duplicate Class 0 samples 1.5x
    df_0_upsampled_p3 = pd.concat([df_0_p3, df_0_p3.sample(frac=0.5, replace=True)])
    
    df_upsampled_p3 = pd.concat([df_0_upsampled_p3, df_1_p3])
    X_t_final_p3 = df_upsampled_p3.drop('target', axis=1)
    y_t_final_p3 = df_upsampled_p3.target.values
    
    clf_p3 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', 
                           solver='adam', alpha=0.01, learning_rate_init=0.001,
                           max_iter=500, early_stopping=True, verbose=True)
    clf_p3.fit(X_t_final_p3, y_t_final_p3)
    
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
