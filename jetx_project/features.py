
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
        # Too early, return empty or default
        # But we can try with what we have
        slice_start = 0
    else:
        slice_start = max(0, current_index - needed_history)
        
    # We include current_index because we might need it for some logic, 
    # but mostly we need data BEFORE current_index.
    # Actually, extract_features_batch calculates features for row 'i' using data up to 'i-1'.
    # So if we want features for 'current_index', we need a DataFrame that includes 'current_index' as a row (even if value is NaN or dummy),
    # so that shift(1) works and brings data from current_index-1.
    
    history_slice = history_full[slice_start : current_index]
    
    # Create DataFrame
    # We add a dummy row at the end to represent 'current_index'
    # The value of this dummy row doesn't matter for shift(1) features, 
    # but it matters if we use current row values (we shouldn't).
    # All our features are shift(1), so they use previous rows.
    
    df_slice = pd.DataFrame({'value': history_slice})
    
    # Append dummy row for the prediction point
    # We can use concat
    dummy_row = pd.DataFrame({'value': [0.0]}) # Value doesn't matter
    df_slice = pd.concat([df_slice, dummy_row], ignore_index=True)
    
    # Now the last index of df_slice corresponds to 'current_index'
    
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
    
    # 6. Long Streak (>=8) Analysis (Vectorized)
    # We need to find the end index of the last streak >= 8.
    
    # Create binary color series (0: Red, 1: Green)
    colors = (values >= 1.5).astype(int)
    
    # Identify runs using diff
    # 1 where change occurs
    change_mask = colors.diff().ne(0)
    # Cumulative sum gives group IDs for consecutive runs
    run_ids = change_mask.cumsum()
    
    # Calculate length of each run
    # Group by run_id and count
    run_lengths = colors.groupby(run_ids).transform('count')
    
    # Mark indices where a long streak ENDS
    # A streak ends at the last index of the group.
    # We want to know if the group it belongs to has length >= 8.
    # And we only care about the end of it.
    
    # Shift run_ids to find end of runs (where next is different)
    # or just use the fact that we have run_lengths aligned.
    
    # We want to find indices 'i' such that:
    # 1. The run ending at 'i' has length >= 8.
    
    # Let's find the end indices of runs.
    # A run ends where the NEXT value is different (change_mask is True at i+1)
    # or it's the last element.
    
    # Actually, simpler:
    # Mark all positions that are part of a long streak
    is_long_streak = (run_lengths >= 8)
    
    # We want "Games Since Last Long Streak ENDED".
    # So we want the index of the last time `is_long_streak` was True AND the streak ended.
    
    # Find end of runs
    # Shift(-1) of run_ids != run_ids
    run_ends = (run_ids != run_ids.shift(-1))
    
    # Mask for "End of a Long Streak"
    long_streak_end_mask = is_long_streak & run_ends
    
    # Now find distance to last True in long_streak_end_mask
    last_long_streak_end_idx = pd.Series(np.where(long_streak_end_mask, df.index, np.nan), index=df.index).ffill().shift(1)
    
    new_features['games_since_long_streak'] = df.index - last_long_streak_end_idx
    new_features['games_since_long_streak'] = new_features['games_since_long_streak'].fillna(200.0)
    
    # Type and Length of that last streak
    # We can fetch values using the index
    # We need to handle NaNs carefully.
    
    # Create a series for type and len at the end index
    streak_types = colors.where(long_streak_end_mask) + 1 # 1: Red, 2: Green
    streak_lens = run_lengths.where(long_streak_end_mask)
    
    # Propagate these values forward
    new_features['last_long_streak_type'] = streak_types.ffill().shift(1).fillna(0.0)
    new_features['last_long_streak_len'] = streak_lens.ffill().shift(1).fillna(0.0)
    
    # 7. Medium Win Streak
    # Games since NOT (1.5 <= val <= 3.0)
    mask_not_med = ~((values >= 1.50) & (values <= 3.00))
    last_not_med_idx = pd.Series(np.where(mask_not_med, df.index, np.nan), index=df.index).ffill().shift(1)
    new_features['medium_win_streak'] = df.index - last_not_med_idx
    new_features['medium_win_streak'] = new_features['medium_win_streak'].fillna(0.0)

    # 8. Teknik İndikatörler (log_values üzerinden)
    delta = log_values.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    new_features['rsi_14'] = (100 - (100 / (1 + rs))).shift(1).fillna(50)

    # Bollinger Bantları (20)
    sma_20 = log_values.rolling(window=20).mean()
    std_20 = log_values.rolling(window=20).std()
    upper_band = sma_20 + (std_20 * 2)
    lower_band = sma_20 - (std_20 * 2)
    percent_b = (log_values - lower_band) / (upper_band - lower_band)
    new_features['bb_percent_b'] = percent_b.shift(1).fillna(0.5)
    new_features['bb_width'] = (upper_band - lower_band).shift(1).fillna(0)

    # MACD (12,26,9)
    ema_12 = log_values.ewm(span=12, adjust=False).mean()
    ema_26 = log_values.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    new_features['macd_hist'] = (macd_line - signal_line).shift(1).fillna(0)

    # 9. Gelişmiş Frekans Özellikleri
    is_over_2 = (values >= 2.00).astype(int)
    new_features['freq_over_2_last_20'] = is_over_2.rolling(20).sum().shift(1).fillna(0)

    is_over_10 = (values >= 10.00).astype(int)
    new_features['freq_over_10_last_50'] = is_over_10.rolling(50).sum().shift(1).fillna(0)

    is_crash = (values <= 1.10).astype(int)
    new_features['freq_crash_last_20'] = is_crash.rolling(20).sum().shift(1).fillna(0)
    
    # Concatenate all new features at once (Optimized)
    df_new = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, df_new], axis=1)
    
    return df
