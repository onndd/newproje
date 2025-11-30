
import numpy as np
import pandas as pd
from .config import WINDOWS, SET1_RANGES, SET2_RANGES, SET3_RANGES
from .categorization import get_set1_id, get_set2_id, get_set3_id

def calculate_streaks(window_values):
    """
    Calculates streak information for a window of values.
    Red: < 1.5
    Green: >= 1.5
    
    Returns:
        longest_red, longest_green, current_red, current_green
    """
    if len(window_values) == 0:
        return 0, 0, 0, 0

    is_green = window_values >= 1.5
    
    # Longest streaks
    max_red = 0
    max_green = 0
    curr_red = 0
    curr_green = 0
    
    for green in is_green:
        if green:
            curr_green += 1
            max_green = max(max_green, curr_green)
            curr_red = 0
        else:
            curr_red += 1
            max_red = max(max_red, curr_red)
            curr_green = 0
            
    # Current streaks (from the end of the window)
    # Reverse iterate to find current streak
    last_streak_red = 0
    last_streak_green = 0
    
    for green in reversed(is_green):
        if green:
            if last_streak_red > 0: # Break if we were counting red
                break
            last_streak_green += 1
        else:
            if last_streak_green > 0: # Break if we were counting green
                break
            last_streak_red += 1
            
    return max_red, max_green, last_streak_red, last_streak_green

def extract_window_features(window_values):
    """
    Extracts features for a single window of data.
    """
    features = {}
    
    # 1. Set1 Distribution
    set1_counts = np.zeros(len(SET1_RANGES))
    for v in window_values:
        idx = get_set1_id(v) - 1 # 0-indexed
        if 0 <= idx < len(set1_counts):
            set1_counts[idx] += 1
    for i, count in enumerate(set1_counts):
        features[f's1_{i+1:02d}'] = count

    # 2. Set2 Distribution
    set2_counts = np.zeros(len(SET2_RANGES))
    for v in window_values:
        idx = get_set2_id(v) - 1
        if 0 <= idx < len(set2_counts):
            set2_counts[idx] += 1
    for i, count in enumerate(set2_counts):
        features[f's2_{i+1:02d}'] = count

    # 3. Set3 Distribution
    set3_counts = np.zeros(len(SET3_RANGES))
    for v in window_values:
        idx = get_set3_id(v) - 1
        if 0 <= idx < len(set3_counts):
            set3_counts[idx] += 1
    for i, count in enumerate(set3_counts):
        features[f's3_{i+1:02d}'] = count

    # 4. Red/Green Counts
    red_count = np.sum(window_values < 1.5)
    green_count = np.sum(window_values >= 1.5)
    features['count_red'] = red_count
    features['count_green'] = green_count

    # 5. Streak Info
    max_red, max_green, cur_red, cur_green = calculate_streaks(window_values)
    features['max_streak_red'] = max_red
    features['max_streak_green'] = max_green
    features['cur_streak_red'] = cur_red
    features['cur_streak_green'] = cur_green

    # 6. Big Multipliers
    features['count_5x'] = np.sum(window_values >= 5.0)
    features['count_10x'] = np.sum(window_values >= 10.0)
    
    return features

def extract_features(history_full, current_index):
    """
    Extracts all features for Model A at a specific point in time (current_index).
    Looks back using defined WINDOWS.
    
    Args:
        history_full: Full array of 'value' history
        current_index: The index for which we want to predict the NEXT value.
    
    Returns:
        A flat dictionary of features.
    """
    all_features = {}
    
    for w_size in WINDOWS:
        if current_index < w_size:
            continue
            
        window = history_full[current_index-w_size:current_index]
        
        # --- Categorical / Pattern Features ONLY ---
        
        # Category Counts (Pattern Structure)
        set1_count = sum(1 for x in window if 1.00 <= x <= 1.49)
        set2_count = sum(1 for x in window if 1.50 <= x <= 1.99)
        set3_count = sum(1 for x in window if x >= 2.00)
        
        w_feats = {
            'set1_ratio': set1_count / w_size,
            'set2_ratio': set2_count / w_size,
            'set3_ratio': set3_count / w_size,
            
            # Streaks (Recent behavior)
            'current_streak_under_2': 0, 
            'current_streak_over_2': 0
        }
        
        # Calculate streaks from the end of window backwards
        curr_streak_u = 0
        curr_streak_o = 0
        for val in reversed(window):
            if val < 2.0:
                if curr_streak_o > 0: break
                curr_streak_u += 1
            else:
                if curr_streak_u > 0: break
                curr_streak_o += 1
                
        w_feats['current_streak_under_2'] = curr_streak_u
        w_feats['current_streak_over_2'] = curr_streak_o

        for k, v in w_feats.items():
            all_features[f'w{w_size}_{k}'] = v
            
    # 3. Raw Numeric History (The Core Feature)
    # Add the actual values of the last 200 games directly
    for lag in range(1, 201): 
        if current_index - lag + 1 >= 0:
            all_features[f'raw_lag_{lag}'] = history_full[current_index - lag + 1]
        else:
            all_features[f'raw_lag_{lag}'] = 0.0
            
    return all_features
