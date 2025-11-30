
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
        current_index: The index for which we want to predict the NEXT value (target is at current_index + 1)
                       So we look at history_full[:current_index+1] effectively? 
                       No, prompt says: "Pencereler: 50: X[i-49..i]" for index i.
                       So if we are at index i, we use data up to i (inclusive).
    
    Returns:
        A flat dictionary or array of features.
    """
    all_features = {}
    
    # Ensure we have enough history for the largest window?
    # Or just pad/handle short history? Prompt implies we start from i=500.
    
    for w_size in WINDOWS:
        start_idx = current_index - w_size + 1
        if start_idx < 0:
            # Not enough data for this window
            # We could return zeros or handle it. 
            # For now, let's assume the caller handles the range start.
            window_data = history_full[0 : current_index + 1] # Take what we have?
            # Or strictly return NaN? 
            # Let's take the slice. If strict:
            # window_data = history_full[start_idx : current_index + 1]
            pass 
        
        # Strict slicing as per prompt X[i-49..i]
        # Python slice: [start : end] includes start, excludes end.
        # So X[i-49..i] (inclusive) -> slice [i-49 : i+1]
        
        slice_start = current_index - w_size + 1
        slice_end = current_index + 1
        
        if slice_start < 0:
             # Fallback for early indices if needed, though training starts later
             window_data = np.array([]) 
        else:
            window_data = history_full[slice_start : slice_end]
            
        w_feats = extract_window_features(window_data)
        
        # Prefix keys with window size
        for k, v in w_feats.items():
            all_features[f'w{w_size}_{k}'] = v
            
    # Add the actual values of the last 10, 20, 50 games directly
    # This helps the tree model see the exact sequence, not just stats
    # We'll add up to the last 50 games, padding with NaN if not enough history
    for lag in range(1, 51): # Last 50 games raw values
        if current_index - lag + 1 >= 0:
            all_features[f'raw_lag_{lag}'] = history_full[current_index - lag + 1]
        else:
            all_features[f'raw_lag_{lag}'] = np.nan # Or 0, depending on desired padding
            
    return all_features

```
