
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
    # Calculate streak (advanced pandas logic)
    # Group by consecutive values
    # For speed, we use a simpler iterative or vector approach for 'current streak'
    # Here using a rolling sum isn't exactly streak, so let's use a specialized loop or optimized vector method
    
    # Vectorized Streak Calculation
    m = is_under_2.astype(bool)
    df['streak_under_2'] = (m.groupby((~m).cumsum()).cumcount())
    df['streak_under_2'] = df['streak_under_2'].shift(1).fillna(0) # We know streak BEFORE current game
    
    # Streak over 2.0x (Hot streak)
    m2 = (values >= 2.0).astype(bool)
    df['streak_over_2'] = (m2.groupby((~m2).cumsum()).cumcount())
    df['streak_over_2'] = df['streak_over_2'].shift(1).fillna(0)

    # 4. Time Since Last High X (Gambler's Fallacy features - models love these)
    # Time since last 10x
    # (Requires cumulative count logic, simplified here)
    # ...
    
    df = df.dropna()
    return df
