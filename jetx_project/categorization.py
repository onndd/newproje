import numpy as np
from .config import SET1_RANGES, SET2_RANGES, SET3_RANGES, SET4_RANGES, SET5_RANGES, SET6_RANGES

# Pre-compute bins for vectorized operations
# We take the lower bound of each range.
_SET1_BINS = np.array([r[0] for r in SET1_RANGES])
_SET2_BINS = np.array([r[0] for r in SET2_RANGES])
_SET3_BINS = np.array([r[0] for r in SET3_RANGES])
_SET4_BINS = np.array([r[0] for r in SET4_RANGES])
_SET5_BINS = np.array([r[0] for r in SET5_RANGES])
_SET6_BINS = np.array([r[0] for r in SET6_RANGES])

def get_ids(values, bins):
    """
    Vectorized category ID retrieval.
    """
    ids = np.digitize(values, bins)
    
    # Handle values smaller than the first bin (return 1)
    if isinstance(ids, np.ndarray):
        ids[ids == 0] = 1
    elif ids == 0:
        ids = 1
        
    return ids

def get_set1_ids(values):
    return get_ids(values, _SET1_BINS)

def get_set2_ids(values):
    return get_ids(values, _SET2_BINS)

def get_set3_ids(values):
    return get_ids(values, _SET3_BINS)

def get_set4_ids(values):
    return get_ids(values, _SET4_BINS)

def get_set5_ids(values):
    return get_ids(values, _SET5_BINS)

def get_set6_ids(values):
    return get_ids(values, _SET6_BINS)

# Legacy support (Singular) - Wraps vectorized version
def get_set1_id(value): return int(get_set1_ids(value))
def get_set2_id(value): return int(get_set2_ids(value))
def get_set3_id(value): return int(get_set3_ids(value))
def get_set4_id(value): return int(get_set4_ids(value))
def get_set5_id(value): return int(get_set5_ids(value))
def get_set6_id(value): return int(get_set6_ids(value))
