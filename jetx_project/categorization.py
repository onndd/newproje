
from .config import SET1_RANGES, SET2_RANGES, SET3_RANGES

def get_category_id(value, ranges):
    """
    Generic function to find the category ID for a given value.
    Returns 1-based index of the matching range.
    Uses INCLUSIVE logic: low <= value <= high
    """
    for i, (low, high) in enumerate(ranges):
        if low <= value <= high:
            return i + 1
    
    # Fallback: if value is somehow between gaps (e.g. 1.495), 
    # we assign it to the closest lower bucket or raise error?
    # For now, let's assume it belongs to the next bucket if it's > high of current?
    # Or just return len(ranges) if it exceeds everything.
    # Given the gaps are 0.01 and data is likely 2 decimals, this loop should catch it.
    
    # If not found (e.g. smaller than 1.00), return 1? Or 0?
    if value < ranges[0][0]:
        return 1
        
    return len(ranges)

def get_set1_id(value):
    """Returns Set1 category ID (1-40)"""
    return get_category_id(value, SET1_RANGES)

def get_set2_id(value):
    """Returns Set2 category ID (1-15)"""
    return get_category_id(value, SET2_RANGES)

def get_set3_id(value):
    """Returns Set3 category ID (1-8)"""
    return get_category_id(value, SET3_RANGES)
