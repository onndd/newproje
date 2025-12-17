
import numpy as np
import pandas as pd
from typing import Dict, Any
from .config import WINDOWS, SET1_RANGES, SET2_RANGES, SET3_RANGES
from .categorization import get_set1_id, get_set2_id, get_set3_id

def extract_features(history_full: np.ndarray, current_index: int) -> Dict[str, float]:
    """
    Extracts all features for Model A at a specific point in time (current_index).
    Optimized: Uses extract_features_batch on a small window.
    """
    # 1. Determine Window Size
    # We need enough history for the largest window + some buffer for rolling/shifting
    # extract_features_batch uses rolling(max_window).shift(1).
    # So we need at least max_window + 1 items.
    # To be safe, let's take max_window + 100.
    
    max_window = max(WINDOWS)
    needed_history = max_window + 100
    
    # 2. Slice History
    # We want the window ending at current_index (exclusive of current_index for the target, 
    # but inclusive for feature calculation which uses past data).
    # Wait, current_index is the index we want to predict FOR.
    # So we have data up to current_index-1.
    # history_full[0...current_index-1] is available.
    
    if current_index < 100:
        slice_start = 0
    else:
        slice_start = max(0, current_index - needed_history)
        
    # Dahil edilecek aralık: slice_start..current_index (current_index HARİÇ)
    # CRITICAL FIX: We must NOT include the target value at current_index in the features.
    # extract_features_batch internally shifts by 1, but to be absolutely safe and logical,
    # the input history should end at current_index-1.
    history_slice = history_full[slice_start : current_index]
    
    # Create DataFrame (gerçek değerlerle, dummy yok)
    df_slice = pd.DataFrame({'value': history_slice})
    
    # 3. Call Batch Extraction
    df_features = extract_features_batch(df_slice)
    
    # 4. Return Last Row
    # The last row contains features for 'current_index'
    last_row = df_features.iloc[-1]
    
    # Convert to dict
    # Filter out non-feature columns if any (like 'value')
    features_dict = last_row.drop('value').to_dict()
    
    return features_dict

def extract_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized feature extraction for the entire DataFrame.
    Much faster than looping extract_features() for training.
    """
    df = df.copy()
    
    # Ensure value is float
    values = df['value'].astype(float)
    log_values = np.log1p(values)
    
    # 1. Rolling Window Features (Set Ratios)
    new_features = {}
    
    for w in WINDOWS:
        # Set 1 (1.00 - 1.49)
        s1 = (values >= 1.00) & (values <= 1.49)
        new_features[f'w{w}_set1_ratio'] = s1.rolling(w).mean().shift(1)
        
        # Set 4 (1.00 - 1.19)
        s4 = (values >= 1.00) & (values <= 1.19)
        new_features[f'w{w}_set4_danger_ratio'] = s4.rolling(w).mean().shift(1)
        
        # Set 5 (1.50 - 1.99)
        s5 = (values >= 1.50) & (values <= 1.99)
        new_features[f'w{w}_set5_safe_ratio'] = s5.rolling(w).mean().shift(1)
        
        # Set 6 (>= 2.00)
        s6 = (values >= 2.00)
        new_features[f'w{w}_set6_high_ratio'] = s6.rolling(w).mean().shift(1)
        
        # Streaks (Vectorized "Games Since")
        # Current Streak Under 2.0 = Games since last >= 2.0
        # We use a cumulative count reset by the condition
        # Trick: Group by cumulative sum of the condition
        
        # Under 2.0 Streak (Reset when val >= 2.0)
        # We want the streak *before* the current row, so we look at shift(1)
        # But rolling logic is easier:
        # Create a series where 1 = break (>= 2.0), 0 = continue (< 2.0)
        # We want distance to last '1'.
        
        # Vectorized "Games Since Last True"
        # 1. Create mask
        mask_u = (values >= 2.0) # The "break" condition
        # 2. Create an array of indices where mask is True
        # We use ffill on indices
        last_break_idx_u = pd.Series(np.where(mask_u, df.index, np.nan), index=df.index).ffill().shift(1)
        new_features[f'w{w}_current_streak_under_2'] = df.index - last_break_idx_u
        # Fill NaNs (start of data) with index (assumes streak started at 0)
        new_features[f'w{w}_current_streak_under_2'] = new_features[f'w{w}_current_streak_under_2'].fillna(df.index.to_series())
        
        # Over 2.0 Streak (Reset when val < 2.0)
        mask_o = (values < 2.0)
        last_break_idx_o = pd.Series(np.where(mask_o, df.index, np.nan), index=df.index).ffill().shift(1)
        new_features[f'w{w}_current_streak_over_2'] = df.index - last_break_idx_o
        new_features[f'w{w}_current_streak_over_2'] = new_features[f'w{w}_current_streak_over_2'].fillna(df.index.to_series())

    # 2. Raw Lags (Vectorized)
    lag_max = 20
    # Create all lag columns at once (sadece yakın geçmişi ham olarak tutuyoruz)
    for i in range(1, lag_max + 1):
        new_features[f'raw_lag_{i}'] = values.shift(i)

    # 3. RTP Balance (Vectorized)
    # (Value - 0.97) rolling sum
    new_features['rtp_balance_500'] = (values - 0.97).rolling(500).sum().shift(1)
    
    # 4. Rolling Median (Noise Filter)
    new_features['rolling_median_20'] = values.rolling(20).median().shift(1)
    
    # 5. Volatility & Chop (Vectorized)
    vol_w = 20
    new_features['volatility_last_20'] = values.rolling(vol_w).std().shift(1)
    new_features['volatility_log_20'] = log_values.rolling(vol_w).std().shift(1)
    
    # Chop Index: Sum of absolute diffs of binary color / window size
    is_green = (values >= 1.5).astype(int)
    changes = is_green.diff().abs()
    new_features['chop_index_20'] = changes.rolling(vol_w).sum().shift(1) / vol_w
    
    # 6. Shockwave (Games Since Big X)
    # Break condition: val >= 10.0
    mask_big = (values >= 10.0)
    last_big_idx = pd.Series(np.where(mask_big, df.index, np.nan), index=df.index).ffill().shift(1) # SAFETY: shift(1) prevents look-ahead
    new_features['games_since_big_x'] = df.index - last_big_idx
    new_features['games_since_big_x'] = new_features['games_since_big_x'].fillna(200) # Default cap
    
    # Last Big X Value
    # We can map the index back to value
    # df['last_big_x_val'] = df['value'].loc[last_big_idx].values # This is tricky with NaNs
    # Easier: ffill the value where mask is true
    new_features['last_big_x_val'] = values.where(mask_big).ffill().shift(1).fillna(0.0)
    
    new_features['is_aftershock'] = (new_features['games_since_big_x'] <= 50).astype(float)
    
    # 6b. Huge Shock (>=100x)
    mask_huge = (values >= 100.0)
    last_huge_idx = pd.Series(np.where(mask_huge, df.index, np.nan), index=df.index).ffill().shift(1)
    new_features['games_since_huge_x'] = (df.index - last_huge_idx).fillna(200)
    
    # 7. Instant Bust Flag (<=1.05)
    new_features['is_instant_bust'] = (values <= 1.05).shift(1).fillna(0.0).astype(float)
    
    # 6. Long Streak (>=8) Analysis (Causal)
    # Fix: Use PREVIOUS game colors to prevent leakage.
    # At time t, we only know the color of t-1.
    is_crash = (values <= 1.10).astype(int)
    new_features['freq_crash_last_20'] = is_crash.rolling(20).sum().shift(1).fillna(0)
    
    # --- NEW ADVANCED FEATURES (Survival & Gambler's Fallacy) ---
    
    # 1. Survival Analysis: P(Reach 2.0 | Reached 1.5)
    # Logic: Of the last 50 games that passed 1.50x, how many made it to 2.00x?
    # If this drops, it means the algorithm is "cutting short".
    reached_15 = (values >= 1.50).astype(float)
    reached_20 = (values >= 2.00).astype(float)
    
    roll_15 = reached_15.rolling(50).sum().shift(1)
    roll_20 = reached_20.rolling(50).sum().shift(1)
    
    # Safe Division
    survival_rate = roll_20 / roll_15.replace(0, 1) # Avoid div/0
    new_features['prob_reach_2_given_1_5'] = survival_rate.fillna(0.5) # Default to 50%

    # 2. RTP Gap (House Debt): Gambler's Fallacy Math
    # Expected Return per Game ~ 0.97x (Theoretical)
    # Over 100 games, expected sum ~ 97.0
    # If actual sum is 50.0, the House is "Hoarding" (Gap +47).
    # If actual sum is 200.0, the House is "Bleeding" (Gap -103).
    
    expected_sum_100 = 97.0
    realized_sum_100 = values.rolling(100).sum().shift(1).fillna(97.0)
    
    # Positive Deficit = House owes us money (Potential Win Wave)
    new_features['rtp_deficit_100'] = expected_sum_100 - realized_sum_100
    
    # 3. Crash Density (Anti-Volatility)
    # Instead of Standard Deviation, count how often we see < 1.10x (Instant Deaths)
    # High Density = Danger Zone
    new_features['density_critical_crash_20'] = (values <= 1.10).rolling(20).sum().shift(1).fillna(0)
    
    # 4. Binary Sequence Matching (Pattern Mining)
    # Encode last 4 games as a binary integer (0-15)
    # 1.50+ = 1, <1.50 = 0
    bin_vals = (values >= 1.50).astype(int)
    
    # Weights for binary conversion: 8, 4, 2, 1
    # Pattern at t is determined by t-4, t-3, t-2, t-1
    # We use shift() to align correct history
    p_code = (bin_vals.shift(4) * 8 + 
              bin_vals.shift(3) * 4 + 
              bin_vals.shift(2) * 2 + 
              bin_vals.shift(1) * 1).fillna(0).astype(int)
    
    new_features['pattern_code_4'] = p_code
    
    # NOTE: Calculating "Historical Win Rate of this Pattern" in vectorized pandas without Lookahead is HARD.
    # Instead, we give the model the 'pattern_code_4' as a Categorical Feature.
    # CatBoost loves categorical features. We keep it as integer for now.
    
    colors = (values >= 1.5).astype(int)
    colors_prev = colors.shift(1).fillna(0) # Default to 0 (Red) for start
    
    # Run breaks on PREVIOUS games
    # If color changes between t-2 and t-1, a run ended at t-2.
    # But for simplicity, we track runs purely experienced in history.
    run_breaks = colors_prev.ne(colors_prev.shift()).cumsum()
    
    # Streak length of the run ending at t-1
    streak_len_prev = colors_prev.groupby(run_breaks).cumcount() + 1
    
    # Did a long streak (>8) END at t-1?
    # This happens if:
    # 1. The run 'streak_len_prev' was reset? No.
    # Correct logic for "Streak Ended":
    # A streak of length L ends at index i if color[i] != color[i-1].
    # But we are working with colors_prev.
    # If colors_prev[t] != colors_prev[t-1], it means the game result (t-1) was different from (t-2).
    # This means the streak that was active up to t-2 has ended.
    
    # Let's reconstruct the report's logic which is safer:
    # We want to identify the index where a long streak finished.
    # We need to know the length of the streak BEFORE it broke.
    # To do this without complex lookback, we can iterate or use a shifted approach.
    
    # Simplified Robust Approach:
    # 1. Identify start of runs on colors_prev.
    # 2. Identify length of run just finished.
    
    # Alternative (Simpler):
    # Just calculate is_long_streak_active on t-1.
    is_long_streak_prev = (streak_len_prev >= 8)
    
    # If it WAS active at t-2, and NOT active at t-1 (because color changed), then it ended.
    # But streak_len_prev accounts for the current run length of colors_prev.
    # If colors_prev changed, streak_len_prev resets to 1.
    # So if streak_len_prev dropped from >=8 to 1, then a long streak ended.
    
    streak_len_prev_lag = streak_len_prev.shift(1).fillna(0)
    long_streak_end = (streak_len_prev_lag >= 8) & (streak_len_prev == 1)
    
    last_long_end_idx = pd.Series(np.where(long_streak_end, df.index, np.nan), index=df.index).ffill()
    new_features['games_since_long_streak'] = (df.index - last_long_end_idx).fillna(200)
    
    # Type and Length of the streak that ended
    # If long_streak_end is True at t, it means the streak ended at t-1 (relative to prev).
    # The COLOR of that streak was colors_prev.shift(1) (aka t-2).
    # The LENGTH was streak_len_prev_lag.
    
    new_features['last_long_streak_type'] = colors_prev.shift(1).where(long_streak_end).ffill().fillna(0)
    new_features['last_long_streak_len'] = streak_len_prev_lag.where(long_streak_end).ffill().fillna(0)
    
    # 7. Medium Win Streak
    # Games since NOT (1.5 <= val <= 3.0)
    mask_not_med = ~((values >= 1.50) & (values <= 3.00))
    last_not_med_idx = pd.Series(np.where(mask_not_med, df.index, np.nan), index=df.index).ffill().shift(1)
    new_features['medium_win_streak'] = df.index - last_not_med_idx
    new_features['medium_win_streak'] = new_features['medium_win_streak'].fillna(0.0)

    # C. Advanced Cooldown Index (RTP Tracking) - REMOVED per user request
    # Only "Volatile Cooldown": Ratio of High wins (>=10x) in last 50 games
    new_features['high_density_50'] = (values >= 10.0).rolling(50).mean().shift(1).fillna(0)
    


    # Concatenate all new features at once (Optimized)
    df_new = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, df_new], axis=1)
    
    return df
