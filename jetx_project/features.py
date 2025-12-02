
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
    
    # Convert history to numpy array for vectorization if it isn't already
    history_arr = np.array(history_full)
    
    for w_size in WINDOWS:
        if current_index < w_size:
            continue
            
        # Slice window (numpy array)
        window = history_arr[current_index-w_size:current_index]
        
        # --- Vectorized Categorical / Pattern Features ---
        
        # Set 1 (1.00 - 1.49)
        set1_mask = (window >= 1.00) & (window <= 1.49)
        set1_count = np.sum(set1_mask)
        
        # Set 4 (1.00 - 1.19)
        set4_mask = (window >= 1.00) & (window <= 1.19)
        set4_count = np.sum(set4_mask)
        
        # Set 5 (1.50 - 1.99)
        set5_mask = (window >= 1.50) & (window <= 1.99)
        set5_count = np.sum(set5_mask)
        
        # Set 6 (>= 2.00)
        set6_mask = (window >= 2.00)
        set6_count = np.sum(set6_mask)
        
        w_feats = {
            'set1_ratio': set1_count / w_size,
            'set4_danger_ratio': set4_count / w_size,
            'set5_safe_ratio': set5_count / w_size,
            'set6_high_ratio': set6_count / w_size,
        }
        
        # --- Vectorized Streak Calculation ---
        # We need the streak at the END of the window.
        # We can flip the window and look for the first change.
        
        window_rev = window[::-1]
        
        # Under 2.0 streak
        # Find indices where value is >= 2.0 (breaking the under streak)
        break_indices_u = np.where(window_rev >= 2.0)[0]
        if len(break_indices_u) > 0:
            curr_streak_u = break_indices_u[0] # Distance to first break
        else:
            curr_streak_u = len(window) # Whole window is streak
            
        # Over 2.0 streak
        # Find indices where value is < 2.0 (breaking the over streak)
        break_indices_o = np.where(window_rev < 2.0)[0]
        if len(break_indices_o) > 0:
            curr_streak_o = break_indices_o[0]
        else:
            curr_streak_o = len(window)
            
        w_feats['current_streak_under_2'] = float(curr_streak_u)
        w_feats['current_streak_over_2'] = float(curr_streak_o)
        
        for k, v in w_feats.items():
            all_features[f'w{w_size}_{k}'] = v
            
    # 3. Raw Numeric History (Vectorized)
    # Extract last 200 values
    lag_max = 200
    
    # Get the slice (reversed)
    if current_index >= lag_max:
        raw_lags = history_arr[current_index-lag_max:current_index][::-1]
    else:
        raw_lags = history_arr[:current_index][::-1]
        
    # Pad if necessary (with NaN)
    if len(raw_lags) < lag_max:
        raw_lags = np.pad(raw_lags, (0, lag_max - len(raw_lags)), constant_values=np.nan)
        
    # Create keys and update dictionary in one go
    # Using dictionary comprehension or zip is faster than loop
    # keys = [f'raw_lag_{i+1}' for i in range(lag_max)]
    # all_features.update(zip(keys, raw_lags))
    
    # Even faster: direct dict creation if possible, but we are updating existing dict
    for i, val in enumerate(raw_lags):
         all_features[f'raw_lag_{i+1}'] = val
            
    # --- 4. Psychological Features (RTP & Shockwave) ---
    
    # A. RTP Tracking (Vectorized)
    rtp_window = 500
    if current_index >= rtp_window:
        rtp_slice = history_arr[current_index-rtp_window:current_index]
        rtp_balance = np.sum(rtp_slice) - (rtp_window * 0.97)
        all_features['rtp_balance_500'] = rtp_balance
    else:
        all_features['rtp_balance_500'] = 0.0
        
    # B. Shockwave Analysis (Vectorized)
    # Find last index >= 10.0
    # Search in last 200
    search_window = 200
    start_search = max(0, current_index - search_window)
    recent_slice = history_arr[start_search:current_index]
    
    big_x_indices = np.where(recent_slice >= 10.0)[0]
    
    if len(big_x_indices) > 0:
        # Last big x index relative to slice start
        last_idx_in_slice = big_x_indices[-1]
        # Absolute index
        last_big_x_abs_idx = start_search + last_idx_in_slice
        
        games_since_big_x = current_index - last_big_x_abs_idx
        all_features['games_since_big_x'] = games_since_big_x
        all_features['last_big_x_val'] = history_arr[last_big_x_abs_idx]
        all_features['is_aftershock'] = 1.0 if games_since_big_x <= 50 else 0.0
    else:
        all_features['games_since_big_x'] = 200.0
        all_features['last_big_x_val'] = 0.0
        all_features['is_aftershock'] = 0.0
            
    # C. Long Streak Analysis (Vectorized-ish)
    # Finding "last streak >= 8" is tricky to fully vectorize without scanning.
    # But we can optimize.
    # We need to find the most recent block of identical colors >= 8.
    
    scan_limit = 200
    start_scan = max(0, current_index - scan_limit)
    scan_slice = history_arr[start_scan:current_index]
    
    # Create binary array: 0 for Red (<1.5), 1 for Green (>=1.5)
    colors = (scan_slice >= 1.5).astype(int)
    
    # Find runs
    # We can use diff to find changes
    # Pad with different value to detect start/end
    # This is still a bit complex to vectorize perfectly for "last >= 8"
    # Let's stick to a simplified fast scan on the slice, which is much faster than full history access
    
    last_streak_end_idx = -1
    last_streak_type = 0
    last_streak_len = 0
    
    # Iterate backwards on the slice (numpy array is fast)
    # Using Numba would be ideal, but standard Python on small slice (200) is okay.
    # We can optimize by grouping.
    
    # Group consecutive elements
    # diff != 0 gives change points
    changes = np.diff(colors)
    change_indices = np.where(changes != 0)[0] + 1
    
    # Add start and end
    change_indices = np.concatenate(([0], change_indices, [len(colors)]))
    
    # Lengths of runs
    run_lengths = np.diff(change_indices)
    run_values = colors[change_indices[:-1]] # Value of each run
    
    # Now we have runs. Look for last one >= 8.
    # Iterate backwards through runs
    found = False
    for i in range(len(run_lengths) - 1, -1, -1):
        if run_lengths[i] >= 8:
            # Found it.
            # Calculate absolute end index.
            # The run ends at change_indices[i+1] (relative to slice)
            end_rel_idx = change_indices[i+1] 
            # Absolute end index (exclusive) -> so the last item was at end_rel_idx - 1
            # In original code, last_streak_end_idx was the index AFTER the streak?
            # "last_streak_end_idx = i + 1" in original loop meant the index where streak broke.
            # So it corresponds to start_scan + end_rel_idx.
            
            last_streak_end_idx = start_scan + end_rel_idx
            last_streak_type = 2 if run_values[i] == 1 else 1 # 1: Red, 2: Green
            last_streak_len = run_lengths[i]
            found = True
            break
            
    if found:
        all_features['games_since_long_streak'] = current_index - last_streak_end_idx
        all_features['last_long_streak_type'] = last_streak_type
        all_features['last_long_streak_len'] = last_streak_len
    else:
        all_features['games_since_long_streak'] = 200.0
        all_features['last_long_streak_type'] = 0.0
        all_features['last_long_streak_len'] = 0.0

    # D. Volatility (Vectorized)
    vol_window = 20
    if current_index >= vol_window:
        vol_slice = history_arr[current_index-vol_window:current_index]
        all_features['volatility_last_20'] = np.std(vol_slice)
        
        # Chop Index
        # Count color changes
        vol_colors = (vol_slice >= 1.5).astype(int)
        changes = np.sum(np.abs(np.diff(vol_colors)))
        all_features['chop_index_20'] = changes / vol_window
    else:
        all_features['volatility_last_20'] = 0.0
        all_features['chop_index_20'] = 0.0
        
    # Medium Win Streak (Vectorized)
    # Consecutive games between 1.50 and 3.00 looking backwards
    # Look at last 50 (safe upper bound for a streak)
    med_limit = 50
    start_med = max(0, current_index - med_limit)
    med_slice = history_arr[start_med:current_index][::-1] # Reversed
    
    med_mask = (med_slice >= 1.50) & (med_slice <= 3.00)
    # Find first False
    first_false = np.where(~med_mask)[0]
    if len(first_false) > 0:
        medium_streak = first_false[0]
    else:
        medium_streak = len(med_slice)
        
    all_features['medium_win_streak'] = float(medium_streak)
    
    return all_features
