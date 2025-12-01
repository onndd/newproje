
from .config import SET1_RANGES, SET2_RANGES, SET3_RANGES, SET4_RANGES, SET5_RANGES, SET6_RANGES

def get_category_id(value, ranges):
    """
    Generic function to find the category ID for a given value.
    Returns 1-based index of the matching range.
    Uses INCLUSIVE logic: low <= value <= high
    """
    for i, (low, high) in enumerate(ranges):
        if low <= value <= high:
            return i + 1
    
    if value < ranges[0][0]:
        return 1
        
    return len(ranges)

def get_set1_id(value):
    """Returns Set1 category ID"""
    return get_category_id(value, SET1_RANGES)

def get_set2_id(value):
    """Returns Set2 category ID"""
    return get_category_id(value, SET2_RANGES)

def get_set3_id(value):
    """Returns Set3 category ID"""
    return get_category_id(value, SET3_RANGES)

def get_set4_id(value):
    """Returns Set4 category ID (Ultra-Fine)"""
    return get_category_id(value, SET4_RANGES)

def get_set5_id(value):
    """Returns Set5 category ID (Medium-Fine)"""
    return get_category_id(value, SET5_RANGES)

def get_set6_id(value):
    """Returns Set6 category ID (Coarse)"""
    return get_category_id(value, SET6_RANGES)
