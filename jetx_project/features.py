
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
    # Add the actual values of the last 200 games directly
    for lag in range(1, 201): 
        if current_index - lag + 1 >= 0:
            all_features[f'raw_lag_{lag}'] = history_full[current_index - lag + 1]
        else:
            all_features[f'raw_lag_{lag}'] = 0.0
            
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
            
    return all_features
