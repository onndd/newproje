
# Configuration for JetX Prediction System

# Window sizes for feature extraction
WINDOWS = [50, 100, 200, 500]

# HMM Binning Configuration
# We want to ensure 1.50x is a hard boundary for one of the bins.
# Previous bins were log-spaced but didn't align with 1.50x.
# New Bins: [1.00, 1.20, 1.49, 2.00, 5.00, 10000.0]
# Bin 0: 1.00 - 1.19 (Very Low)
# Bin 1: 1.20 - 1.49 (Low - Loss Zone)
# Bin 2: 1.50 - 1.99 (Target Zone)
# Bin 3: 2.00 - 4.99 (High)
# Bin 4: 5.00+ (Very High)
# HMM Binning Configuration
# Critical Update: Finer granularity in the <1.50 danger zone.
# This forces the HMM to distinguish between "Instant Death" (1.0-1.1) and "Near Miss" (1.3-1.4).
# New Bins: [1.00, 1.10, 1.20, 1.35, 1.49, 1.99, 10000.0]
# Bin 0: 1.00 - 1.09 (Extreme Risk)
# Bin 1: 1.10 - 1.19 (High Risk)
# Bin 2: 1.20 - 1.34 (Medium Risk)
# Bin 3: 1.35 - 1.49 (Low Risk / Pre-Target)
# Bin 4: 1.50 - 1.99 (Target Zone)
# Bin 5: 2.00+ (Bonus Zone)
HMM_BIN_EDGES = [1.00, 1.10, 1.20, 1.35, 1.49, 1.99, 10000.0]

# NOTE: Ranges are now defined as INCLUSIVE [lower, upper].
# We assume data has 2 decimal places.
# Example: (1.00, 1.14) includes 1.00 and 1.14.
# The next category starts at 1.15.

# Set 1 Categories (Fine)
SET1_RANGES = [
    (1.00, 1.14), (1.15, 1.24), (1.25, 1.34), (1.35, 1.49), (1.50, 1.64),
    (1.65, 1.79), (1.80, 1.94), (1.95, 2.09), (2.10, 2.24), (2.25, 2.39),
    (2.40, 2.54), (2.55, 2.69), (2.70, 2.84), (2.85, 2.99), (3.00, 3.14),
    (3.15, 3.29), (3.30, 3.44), (3.45, 3.59), (3.60, 3.74), (3.75, 3.89),
    (3.90, 4.04), (4.05, 4.19), (4.20, 4.34), (4.35, 4.49), (4.50, 4.64),
    (4.65, 4.79), (4.80, 4.94), (4.95, 4.99), (5.00, 5.99), (6.00, 7.49),
    (7.50, 9.99), (10.00, 12.99), (13.00, 15.99), (16.00, 19.99),
    (20.00, 29.99), (30.00, 39.99), (40.00, 49.99), (50.00, 69.99),
    (70.00, 99.99), (100.00, 199.99), (200.00, 499.99), 
    (500.00, 999.99), (1000.00, 1999.99), (2000.00, 2999.99), (3000.00, float('inf'))
]

# Set 2 Categories (Medium)
SET2_RANGES = [
    (1.00, 1.29), (1.30, 1.49), (1.50, 1.99), (2.00, 2.49), (2.50, 2.99),
    (3.00, 3.99), (4.00, 4.99), (5.00, 7.49), (7.50, 9.99), (10.00, 14.99),
    (15.00, 19.99), (20.00, 29.99), (30.00, 49.99), (50.00, 99.99),
    (100.00, 199.99), (200.00, 499.99), 
    (500.00, 999.99), (1000.00, 1999.99), (2000.00, 2999.99), (3000.00, float('inf'))
]

# Set 3 Categories (Coarse)
SET3_RANGES = [
    (1.00, 1.49), (1.50, 1.99), (2.00, 2.99), (3.00, 4.99), (5.00, 9.99),
    (10.00, 19.99), (20.00, 49.99), (50.00, 99.99), (100.00, 199.99), (200.00, 499.99), 
    (500.00, 999.99), (1000.00, 1999.99), (2000.00, 2999.99), (3000.00, float('inf'))
]

# Set 4 (Ultra-Fine Low Band + Detailed Tail)
SET4_RANGES = [
    (1.00, 1.09), (1.10, 1.14), (1.15, 1.19), (1.20, 1.24), (1.25, 1.29), 
    (1.30, 1.34), (1.35, 1.39), (1.40, 1.44), (1.45, 1.49),
    (1.50, 1.59), (1.60, 1.69), (1.70, 1.79), (1.80, 1.89), (1.90, 1.99),
    (2.00, 2.49), (2.50, 2.99), (3.00, 3.99), (4.00, 4.99), (5.00, 6.99), 
    (7.00, 8.99), (9.00, 11.99), (12.00, 14.99), (15.00, 19.99), 
    (20.00, 29.99), (30.00, 39.99), (40.00, 49.99), (50.00, 69.99), 
    (70.00, 99.99), (100.00, 199.99), (200.00, 499.99), 
    (500.00, 999.99), (1000.00, 1999.99), (2000.00, 2999.99), (3000.00, float('inf'))
]

# Set 5 (Medium Detail + Tail Compression)
SET5_RANGES = [
    (1.00, 1.19), (1.20, 1.29), (1.30, 1.39), (1.40, 1.49),
    (1.50, 1.74), (1.75, 1.99), (2.00, 2.49), (2.50, 2.99), (3.00, 3.99), 
    (4.00, 4.99), (5.00, 6.99), (7.00, 9.99), (10.00, 14.99), (15.00, 19.99), 
    (20.00, 29.99), (30.00, 49.99), (50.00, 79.99), (80.00, 119.99), 
    (120.00, 199.99), (200.00, 499.99), 
    (500.00, 999.99), (1000.00, 1999.99), (2000.00, 2999.99), (3000.00, float('inf'))
]

# Set 6 (Coarse / Fast Test)
SET6_RANGES = [
    (1.00, 1.49), (1.50, 1.99), (2.00, 2.99), (3.00, 4.99), (5.00, 9.99), 
    (10.00, 19.99), (20.00, 39.99), (40.00, 79.99), (80.00, 149.99), 
    (150.00, 499.99), (500.00, 999.99), (1000.00, 1999.99), (2000.00, 2999.99), (3000.00, float('inf'))
]

DB_PATH = 'jetx.db'

# Central DB read limit to avoid OOM across scripts
# DB_LIMIT = 2000 # Eski limit (Veri körlüğüne sebep oluyordu)
DB_LIMIT = 50000 # Yeni limit (Yeterli veri için)

MODEL_DIR = 'models'

# -------------------------------------------------------------------
# PROFIT SCORING WEIGHTS (The "Sniper Logic" Configuration)
# -------------------------------------------------------------------
# Used by optimization.py and model training to calculate expected profit score.
# -------------------------------------------------------------------
# PROFIT SCORING WEIGHTS (The "Sniper Logic" Configuration)
# -------------------------------------------------------------------
# Used by optimization.py and model training to calculate expected profit score.

# DEFAULT / FALLBACK WEIGHTS
PROFIT_SCORING_WEIGHTS = {
    'TP': 100,  # Correct prediction
    'TN': 1,    # Small reward for correct passivity
    'FP': 120,  # Standard penalty for False Alarm
    'FN': 10,   # Missed Opportunity penalty
    'PRECISION': 50 # Bonus for high precision
}

# 1. CATBOOST (Model A - The Sniper)
# Needs to be very precise but not paralyzed.
SCORING_CATBOOST = {
    'TP': 85,
    'TN': 1,
    'FP': 110, # Reduced from 140 to encourage taking more calculated risks
    'FN': 20,  # Slightly higher FOMO penalty to prevent over-silence
    'PRECISION': 65 # High precision bonus
}

# 2. LSTM (Model C - The Pattern Seeker)
# Deep learning can be noisy, allow slightly more leeway but reward catching trends.
SCORING_LSTM = {
    'TP': 90, # Higher reward for catching complex time-series patterns
    'TN': 1,
    'FP': 120, # Lower penalty than CatBoost (DL needs room to breathe)
    'FN': 10,
    'PRECISION': 50
}

# 3. LIGHTGBM (Model D - The Fast Learner)
# Similar to CatBoost but often more aggressive.
SCORING_LIGHTGBM = {
    'TP': 90,
    'TN': 1,
    'FP': 130, # Stricter penalty to curb LightGBM's tendency to over-predict
    'FN': 10,
    'PRECISION': 65
}

# 4. MLP (Model E - The Neural Net)
# Often unstable on tabular data, needs strict guidance.
SCORING_MLP = {
    'TP': 120,
    'TN': 1,
    'FP': 150, # High penalty to prevent noise fitting
    'FN': 5,
    'PRECISION': 50
}

# 5. MEMORY (Model B - The Historian)
# Based on exact matches, usually high precision naturally.
SCORING_MEMORY = {
    'TP': 100,
    'TN': 1,
    'FP': 200, # Very strict! Memory should only speak if it REMEMBERS correctly.
    'FN': 0,   # No penalty for silence (it's okay to not strictly match)
    'PRECISION': 65
}

# 6. TRANSFORMER (Model F - The Visionary)
# Long context, huge potential but risky.
SCORING_TRANSFORMER = {
    'TP': 95,
    'TN': 1,
    'FP': 125,
    'FN': 10,
    'PRECISION': 50
}

# P3.0 Specific Weights (For High Multiplies)
PROFIT_SCORING_WEIGHTS_P3 = {
    'TP': 400,  # HUGE Reward for catching a 3.00x
    'TN': 1,
    'FP': 100,  # Reduced from 150 to 100 to encourage catching rare events
    'FN': 50,   # Standard miss penalty
    'PRECISION': 100
}
