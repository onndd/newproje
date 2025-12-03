
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest

def extract_anomaly_features(values, window_size=300):
    """
    Extracts specific features for Anomaly Detection.
    Features:
    1. RTP (Return to Player) of last 50 games.
    2. Volatility (Std Dev) of last 50 games.
    3. Consecutive Red Series Length (Current).
    """
    # We need to calculate these for every point in time to train
    # or just for the current window to predict.
    
    # If values is a long array (training), we return a DataFrame of features
    s_values = pd.Series(values)
    
    # 1. RTP (Last 50)
    # RTP = Sum(Values) / (Count * 0.97)? No, RTP is usually Payout / Bet.
    # Here we approximate "Market RTP" as Average Multiplier? 
    # Or Balance: (Sum(Values) - Count * 0.97)
    # Let's use the definition from features.py: (Value - 0.97) rolling sum.
    # But normalized by window size to be comparable?
    # Let's use Rolling Mean of (Value - 0.97)
    rtp_50 = (s_values - 0.97).rolling(window_size).mean()
    
    # 2. Volatility (Last 50)
    vol_50 = s_values.rolling(window_size).std()
    
    # 3. Consecutive Red Series Length
    # Red = Value < 1.50
    is_red = (s_values < 1.50)
    # Group by consecutive values
    # We want "Current Red Streak" at each point.
    # If is_red is False, streak is 0.
    # If is_red is True, streak is 1 + prev_streak.
    
    # Vectorized Streak Calculation
    # Create groups of consecutive values
    # cumsum of (val != prev_val)
    groups = is_red.ne(is_red.shift()).cumsum()
    # cumcount gives 0, 1, 2... for each group
    streaks = is_red.groupby(groups).cumsum() # This sums boolean? No.
    # We want count.
    # Actually:
    # If we group by 'groups', we can get the running count?
    # Simpler:
    # streak = s.groupby((s != s.shift()).cumsum()).cumcount() + 1
    # Then multiply by is_red to zero out non-reds.
    
    streak_counter = s_values.groupby((is_red != is_red.shift()).cumsum()).cumcount() + 1
    red_streak = streak_counter * is_red.astype(int)
    
    # Combine
    features = pd.DataFrame({
        'rtp_50': rtp_50,
        'vol_50': vol_50,
        'red_streak': red_streak
    })
    
    # Drop NaNs (first 50)
    features = features.dropna()
    
    return features

def train_anomaly_detector(values):
    """
    Trains the Isolation Forest model.
    """
    print("Training Anomaly Detector (The Shield)...")
    
    # Extract features
    features = extract_anomaly_features(values)
    
    if len(features) < 100:
        print("Warning: Not enough data for Anomaly Detection.")
        return None
        
    # Isolation Forest
    # contamination='auto' or low value (e.g. 0.01) assuming anomalies are rare
    clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    clf.fit(features)
    
    return clf

def check_anomaly(model, current_window_values):
    """
    Checks if the current market state is anomalous.
    Returns:
        score: -1 (Anomaly) or 1 (Normal)
        details: Dict with feature values
    """
    if model is None:
        return 1, {}
        
    # We need at least 50 values
    if len(current_window_values) < 50:
        return 1, {}
        
    # Extract features for the LAST point only
    # We can pass the whole window and take the last row
    features_df = extract_anomaly_features(current_window_values)
    
    if len(features_df) == 0:
        return 1, {}
        
    last_features = features_df.iloc[[-1]] # Keep as DataFrame
    
    # Predict
    score = model.predict(last_features)[0]
    
    details = last_features.iloc[0].to_dict()
    
    return score, details

def save_anomaly_detector(model, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    joblib.dump(model, os.path.join(output_dir, 'model_anomaly.pkl'))
    print(f"Anomaly Detector saved to {output_dir}")

def load_anomaly_detector(model_dir='.'):
    path = os.path.join(model_dir, 'model_anomaly.pkl')
    if not os.path.exists(path):
        return None
    return joblib.load(path)
