
import pandas as pd
import numpy as np

def extract_features(df, windows=[10, 50, 100]):
    """
    Extracts purely numerical features for LightGBM.
    Focuses on: Moving Averages, Volatility, Streaks, Last N values.
    """
    df = df.copy()
    values = df['value']
    
    # 1. Lag Features (Last 10 results)
    for i in range(1, 11):
        df[f'lag_{i}'] = values.shift(i)
        
    # 2. Rolling Statistics
    for w in windows:
        rolled = values.shift(1).rolling(window=w)
        df[f'rol_mean_{w}'] = rolled.mean()
        df[f'rol_std_{w}'] = rolled.std()
        df[f'rol_max_{w}'] = rolled.max()
        # Relative strength (Are we above or below recent mean?)
        df[f'rel_str_{w}'] = (values.shift(1) - df[f'rol_mean_{w}']) / (df[f'rol_std_{w}'] + 1e-9)

    # 3. Crash / Boom Streaks
    # Streak under 2.0x
    is_under_2 = (values < 2.0).astype(int)
    m = is_under_2.astype(bool)
    df['streak_under_2'] = (m.groupby((~m).cumsum()).cumcount())
    df['streak_under_2'] = df['streak_under_2'].shift(1).fillna(0)
    
    # Streak over 2.0x
    m2 = (values >= 2.0).astype(bool)
    df['streak_over_2'] = (m2.groupby((~m2).cumsum()).cumcount())
    df['streak_over_2'] = df['streak_over_2'].shift(1).fillna(0)

    # --- ADVANCED FEATURES (The Hunter's Logic) ---

    # 4. Volatility Squeeze (The Silence)
    # Ratio of Short-term Volatility (10) to Long-term Volatility (100)
    # < 1.0 means quiet, < 0.5 means VERY quiet (Storm coming)
    vol_short = values.shift(1).rolling(10).std()
    vol_long = values.shift(1).rolling(100).std()
    df['vol_squeeze'] = (vol_short / (vol_long + 1e-9)).fillna(1.0)

    # 5. RTP Gap (House Hoarding)
    # Theory: If recent payout < Long term average, House is 'hoarding'
    # Simple proxies: Global Mean vs Rolling Mean 100
    global_mean = values.shift(1).expanding().mean()
    local_mean = values.shift(1).rolling(100).mean()
    df['rtp_gap'] = (global_mean - local_mean).fillna(0) 
    # Positive = House Hoarding (Global > Local) -> Expect High X
    # Negative = House Bleeding (Global < Local) -> Expect Correction

    # 6. Games Since High X (Time Pressure)
    # Vectorized calculation for 'Games Since' specific multipliers
    for threshold in [10.0, 20.0, 50.0]:
        # Create a mask where value >= threshold
        hit_mask = (values >= threshold)
        # Cumulative sum of hits increases by 1 each time hit occurs
        # Group by this sum to isolate segments between hits
        # Cumcount gives count within each segment (0, 1, 2... since last hit)
        # Shift 1 to represent 'entering' the next game
        # Note: We need a slight trick because GroupBy cumcount resets On the hit, not After.
        # Efficient approach:
        last_hit_idx = pd.Series(np.where(hit_mask, df.index, np.nan)).ffill().shift(1)
        df[f'games_since_{int(threshold)}x'] = df.index - last_hit_idx
        df[f'games_since_{int(threshold)}x'] = df[f'games_since_{int(threshold)}x'].fillna(999) # 999 if never seen

    # 7. Momentum Derivative (U-Turn Detect)
    # Rate of change of the trend.
    # Trend = Mean(10). Vel = Diff(Trend). Acc = Diff(Vel).
    trend_10 = values.shift(1).rolling(10).mean()
    velocity = trend_10.diff()
    acceleration = velocity.diff()
    df['trend_acceleration'] = acceleration.fillna(0)
    # If Velocity is Negative (Dropping) but Acceleration is Positive (Slowing down), it's a U-Turn.

    df = df.dropna()
    return df
