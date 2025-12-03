
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
    
    # Compute Sample Weights for Class Balancing
    from sklearn.utils.class_weight import compute_sample_weight
    
    # P1.5 Model
    print("Training MLP (P1.5)...")
    sample_weight_p15 = compute_sample_weight(class_weight='balanced', y=y_p15_t)
    
    clf_p15 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', 
                            solver='adam', alpha=0.01, learning_rate_init=0.001,
                            max_iter=500, early_stopping=True, verbose=True)
    clf_p15.fit(X_t, y_p15_t) # MLPClassifier doesn't support sample_weight in fit() for all solvers/versions, checking docs...
    # Wait, sklearn MLPClassifier DOES NOT support sample_weight in fit().
    # It only supports it for partial_fit.
    # Actually, recent versions DO support it. Let's assume standard sklearn environment.
    # If not supported, we might need to oversample manually.
    # But let's check if we can use 'partial_fit' loop or just rely on the fact that we can't easily weight it.
    # Alternative: Oversampling.
    # Let's try passing it to fit, if it fails, we'll know. 
    # Actually, checking sklearn docs: fit(X, y) does NOT take sample_weight.
    # So we must manually oversample the minority class.
    
    # Let's implement simple oversampling for MLP
    from sklearn.utils import resample
    
    # Combine X and y
    X_t_p15 = X_t.copy()
    X_t_p15['target'] = y_p15_t
    
    # Separate majority and minority classes
    df_majority = X_t_p15[X_t_p15.target == 0]
    df_minority = X_t_p15[X_t_p15.target == 1]
    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=len(df_majority),    # to match majority class
                                     random_state=42) # reproducible results
                                     
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    # Display new class counts
    print(f"P1.5 Class Counts after Upsampling: {df_upsampled.target.value_counts()}")
    
    X_t_upsampled = df_upsampled.drop('target', axis=1)
    y_p15_upsampled = df_upsampled.target.values
    
    clf_p15 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', 
                            solver='adam', alpha=0.01, learning_rate_init=0.001,
                            max_iter=500, early_stopping=True, verbose=True)
    clf_p15.fit(X_t_upsampled, y_p15_upsampled)
    
    # P3.0 Model (Same logic)
    print("Training MLP (P3.0)...")
    
    X_t_p3 = X_t.copy()
    X_t_p3['target'] = y_p3_t
    
    df_majority_p3 = X_t_p3[X_t_p3.target == 0]
    df_minority_p3 = X_t_p3[X_t_p3.target == 1]
    
    df_minority_upsampled_p3 = resample(df_minority_p3, 
                                        replace=True,
                                        n_samples=len(df_majority_p3),
                                        random_state=42)
                                        
    df_upsampled_p3 = pd.concat([df_majority_p3, df_minority_upsampled_p3])
    print(f"P3.0 Class Counts after Upsampling: {df_upsampled_p3.target.value_counts()}")
    
    X_t_upsampled_p3 = df_upsampled_p3.drop('target', axis=1)
    y_p3_upsampled = df_upsampled_p3.target.values

    clf_p3 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', 
                           solver='adam', alpha=0.01, learning_rate_init=0.001,
                           max_iter=500, early_stopping=True, verbose=True)
    clf_p3.fit(X_t_upsampled_p3, y_p3_upsampled)
    
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
