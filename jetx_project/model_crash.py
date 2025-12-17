
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score, precision_score
import joblib
import os
from .config import MODEL_DIR

def train_crash_detector(X, y_crash, model_name="Crash_Guard"):
    """
    Trains an Isolation Forest to detect 'Safe' vs 'Unsafe'.
    
    STRATEGY CHANGE:
    Instead of supervised classification (which failed due to imbalance),
    we use UNSUPERVISED ANOMALY DETECTION on the 'SAFE' data.
    
    1. We define 'Safe' as NOT Crash (value >= 1.30).
    2. We train Isolation Forest ONLY on these 'Safe' examples.
    3. It learns what 'Normal' looks like.
    4. Anything it marks as -1 (Outlier) is a 'Potential Crash'.
    """
    print(f"\n--- Training {model_name} (Isolation Forest - Anomaly Mode) ---")
    
    # 1. Select only SAFE examples for training
    # y_crash = 1 means CRASH (< 1.30).
    # y_crash = 0 means SAFE (>= 1.30).
    mask_safe = (y_crash == 0)
    X_safe = X[mask_safe]
    
    print(f"Training on {len(X_safe)} SAFE examples (ignoring {np.sum(y_crash)} crashes for training)...")
    
    # 2. Train Isolation Forest
    # contamination: We expect roughly how many outliers in test?
    # Actually, we don't set contamination if we want it to define its own boundary,
    # but 'auto' is usually good. Or better: set low contamination (0.05) to be loose,
    # or high to be strict.
    # Let's use 'auto' to let it decide based on tree depth.
    
    clf = IsolationForest(
        n_estimators=300,
        max_samples='auto',
        contamination=0.10, # Assuming top 10% risky of "Safe" might be outliers too? 
        # Actually standard IF assumes contamination in train set.
        # But our train set is PURE safe (theoretically).
        # So we should use contamination very low? 
        # No, Isolation Forest is robust.
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_safe)
    
    # 3. Evaluation (On Full Set)
    # Predict all
    preds_if = clf.predict(X) 
    # IF returns: 1 = Normal (Safe), -1 = Anomaly (Crash)
    
    # Map back to our logic:
    # IF(1) -> Safe (0)
    # IF(-1) -> Crash (1)
    preds_mapped = np.where(preds_if == 1, 0, 1)
    
    rec = recall_score(y_crash, preds_mapped, zero_division=0)
    prec = precision_score(y_crash, preds_mapped, zero_division=0)
    
    print(f"Isolation Forest Results on Training Set:")
    print(f"Caught Crashes (Recall): {rec:.2%}")
    print(f"False Alarms (Precision): {prec:.2%}")
    print(f"Total Flags: {np.sum(preds_mapped)}")
    
    # Save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    save_path = os.path.join(MODEL_DIR, f'{model_name}.joblib')
    joblib.dump(clf, save_path)
    print(f"Crash Guard (IF) saved to {save_path}")
    
    return clf

def load_crash_detector(model_name="Crash_Guard"):
    path = os.path.join(MODEL_DIR, f'{model_name}.joblib')
    if not os.path.exists(path):
        # Fallback to look for old name if needed, but better fail
        raise FileNotFoundError(f"Crash Guard model not found at {path}")
    return joblib.load(path)

def predict_crash(model, X):
    """
    Returns Crash Risk Score.
    Isolation Forest returns 'decision_function' where lower = more anomalous.
    We need to convert this to a pseudo-probability [0, 1].
    
    IF Output:
    > 0  : Normal (Safe)
    < 0  : Anomaly (Crash)
    
    We want:
    1.0 = Definite Crash
    0.0 = Definite Safe
    """
    raw_scores = model.decision_function(X)
    
    # Transform raw scores [-0.5, 0.5] approx
    # Negative is crash.
    # We invert sign so + is crash.
    crash_scores = -raw_scores
    
    # Normalize via Sigmoid-ish or MinMax logic?
    # Simple heuristic:
    # If raw < 0 (crash), we want prob > 0.5
    # If raw > 0 (safe), we want prob < 0.5
    
    # Standard probability conversion for IF scores doesn't exist perfectly.
    # We use MinMax scaler approach roughly based on expected range [-0.2, 0.2]
    
    # Using a simple sigmoid on the inverted score
    probs = 1 / (1 + np.exp(-10 * crash_scores)) # Multiplier 10 makes transition sharp around 0
    
    return probs
