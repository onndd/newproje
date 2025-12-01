
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
        # We use the new Sets 4, 5, 6 for finer granularity
        
        # Set 1 (Fine)
        set1_count = sum(1 for x in window if 1.00 <= x <= 1.49)
        
        # Set 4 (Ultra-Fine Low Band) - Critical for "Harvest Mode"
        # Range: 1.00 - 1.19 (Danger Zone)
        set4_danger_count = sum(1 for x in window if 1.00 <= x <= 1.19)
        
        # Set 5 (Medium Detail)
        # Range: 1.50 - 1.99 (Safe Zone?)
        set5_safe_count = sum(1 for x in window if 1.50 <= x <= 1.99)
        
        # Set 6 (Coarse) - High Multipliers
        # Range: >= 2.00
        set6_high_count = sum(1 for x in window if x >= 2.00)
        
        w_feats = {
            'set1_ratio': set1_count / w_size,
            'set4_danger_ratio': set4_danger_count / w_size,
            'set5_safe_ratio': set5_safe_count / w_size,
            'set6_high_ratio': set6_high_count / w_size,
            
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
        
        # Add Set 4, 5, 6 IDs of the *last* game specifically?
        # No, raw lags cover that.
        
        for k, v in w_feats.items():
            all_features[f'w{w_size}_{k}'] = v
            
    # 3. Raw Numeric History (The Core Feature)
    # 3. Raw Numeric History (The Core Feature)
    # Add the actual values of the last 200 games directly
    # Safety Check: If we don't have enough history, we should ideally skip this sample.
    # However, since start_index is usually 500, we are safe.
    # If called with low index, we return 0.0 which is "safe" padding but technically noise.
    for lag in range(1, 201): 
        if current_index - lag + 1 >= 0:
            all_features[f'raw_lag_{lag}'] = history_full[current_index - lag + 1]
        else:
            # Padding with 1.0 (minimum crash) is better than 0.0 for multipliers
            all_features[f'raw_lag_{lag}'] = 1.0
            
    # --- 4. Psychological Features (RTP & Shockwave) ---
    
    # A. RTP Tracking (Virtual Bankroll)
    # Calculate "Casino Net Profit" over the last 500 games
    # Assumption: 1 unit bet on every game. 
    # If Result < 1.00 (Crash), Casino keeps 1 unit.
    # If Result >= 1.00, Casino pays (Result - 1) units? 
    # Wait, in JetX, if you don't cash out, you lose. 
    # But we don't know when players cash out.
    # Simplification: We track the "Potential Payout" vs "Input".
    # Let's track the average multiplier. If Avg > 1.00, players *could* be winning.
    # Better metric: "Theoretical RTP Balance".
    # If we sum (Multiplier - 0.97), we see if the game is paying above or below theoretical RTP.
    
    rtp_window = 500
    if current_index >= rtp_window:
        rtp_slice = history_full[current_index-rtp_window:current_index]
        # Calculate deviation from expected return (e.g. 0.97)
        # If sum(rtp_slice) is high, the game has been generous -> Expect correction (Cold)
        # If sum(rtp_slice) is low, the game has been stingy -> Expect correction (Hot)
        rtp_balance = np.sum(rtp_slice) - (rtp_window * 0.97) 
        all_features['rtp_balance_500'] = rtp_balance
    else:
        all_features['rtp_balance_500'] = 0.0
        
    # B. Shockwave Analysis (Big X Context)
    # Find the last "Big X" (e.g. >= 10.0)
    last_big_x_idx = -1
    for i in range(current_index - 1, max(-1, current_index - 200), -1):
        if history_full[i] >= 10.0:
            last_big_x_idx = i
            break
            
    if last_big_x_idx != -1:
        games_since_big_x = current_index - last_big_x_idx
        all_features['games_since_big_x'] = games_since_big_x
        all_features['last_big_x_val'] = history_full[last_big_x_idx]
        
        # "Aftershock" phase: Are we in the danger zone?
        all_features['is_aftershock'] = 1 if games_since_big_x <= 50 else 0
    else:
        all_features['games_since_big_x'] = 200 # Max cap
        all_features['last_big_x_val'] = 0.0
        all_features['is_aftershock'] = 0
            
    # C. Long Streak Analysis (Context)
    # Find the last "Long Streak" (e.g. >= 8 games of same color)
    # We look back to find a sequence of >= 8 reds (<1.5) or greens (>=1.5)
    
    last_streak_end_idx = -1
    last_streak_type = 0 # 0: None, 1: Red (<1.5), 2: Green (>=1.5)
    last_streak_len = 0
    
    # Scan backwards from current_index
    # This can be slow if we scan too far, limit to 200
    scan_limit = 200
    
    # Helper to check streak at specific index
    # We need to find where a streak *ended*. 
    # A streak ends at i if i is different from i-1, or if we are just scanning backwards.
    # Actually, we just need to find the *most recent* completed or ongoing streak of length >= 8.
    
    # Let's iterate backwards and count streaks.
    temp_streak_len = 0
    temp_streak_type = 0 # 1: Red, 2: Green
    
    for i in range(current_index - 1, max(-1, current_index - scan_limit), -1):
        val = history_full[i]
        val_type = 1 if val < 1.5 else 2
        
        if temp_streak_len == 0:
            temp_streak_type = val_type
            temp_streak_len = 1
        elif val_type == temp_streak_type:
            temp_streak_len += 1
        else:
            # Streak broke. Was the previous one long enough?
            if temp_streak_len >= 8:
                last_streak_end_idx = i + temp_streak_len # The index where it ended (actually i+1 was the break)
                # Wait, i is the index of the *new* color. So the streak was from i+1 to i+temp_streak_len.
                # The "end" index (most recent part) is i + 1 + temp_streak_len - 1 = i + temp_streak_len?
                # Let's simplify: The streak was active at indices [i+1, i+temp_streak_len].
                # The most recent game of that streak was at i+1.
                last_streak_end_idx = i + 1 
                last_streak_type = temp_streak_type
                last_streak_len = temp_streak_len
                break
            
            # Reset for new streak
            temp_streak_type = val_type
            temp_streak_len = 1
            
    # Check if the loop finished with a long streak (at the very beginning of scan)
    if last_streak_end_idx == -1 and temp_streak_len >= 8:
        last_streak_end_idx = max(-1, current_index - scan_limit) # Approximate
        last_streak_type = temp_streak_type
        last_streak_len = temp_streak_len
         
    if last_streak_end_idx != -1:
        games_since_streak = current_index - last_streak_end_idx
        all_features['games_since_long_streak'] = games_since_streak
        all_features['last_long_streak_type'] = last_streak_type
        all_features['last_long_streak_len'] = last_streak_len
    else:
        all_features['games_since_long_streak'] = 200
        all_features['last_long_streak_type'] = 0
        all_features['last_long_streak_len'] = 0

    # D. Volatility & "Fake High X" Detection
    # Pattern: 
    # 1. Low Volatility (Tease): Many 1.5x - 3.0x wins.
    # 2. High Volatility (Shakeout): Alternating wins/losses.
    # 3. Payoff: Big X.
    
    # 1. Volatility (Std Dev of last 20 games)
    vol_window = 20
    if current_index >= vol_window:
        vol_slice = history_full[current_index-vol_window:current_index]
        all_features['volatility_last_20'] = np.std(vol_slice)
        
        # 2. Chop Index (Alternating Pattern)
        # Count how many times the color changed in the last 20 games
        # Color change: (val_i < 1.5) != (val_i-1 < 1.5)
        chop_count = 0
        for i in range(1, len(vol_slice)):
            prev_color = 1 if vol_slice[i-1] < 1.5 else 2
            curr_color = 1 if vol_slice[i] < 1.5 else 2
            if prev_color != curr_color:
                chop_count += 1
        all_features['chop_index_20'] = chop_count / vol_window
    else:
        all_features['volatility_last_20'] = 0.0
        all_features['chop_index_20'] = 0.0
        
    # 3. Medium Win Streak (The Tease)
    # Count consecutive games between 1.50 and 3.00
    # This indicates a "Safe" period that might precede a storm or a big win.
    medium_streak = 0
    for i in range(current_index - 1, -1, -1):
        val = history_full[i]
        if 1.50 <= val <= 3.00:
            medium_streak += 1
        else:
            break
    all_features['medium_win_streak'] = medium_streak
    
    return all_features
