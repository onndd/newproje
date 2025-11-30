
import numpy as np
import pandas as pd
from .config import WINDOWS, SET1_RANGES, SET2_RANGES, SET3_RANGES
from .categorization import get_set1_id, get_set2_id, get_set3_id



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
