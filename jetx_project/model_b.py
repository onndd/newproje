
import numpy as np
import joblib
import os
from sklearn.neighbors import NearestNeighbors
from .categorization import get_set1_id, get_set2_id, get_set3_id, get_set4_id, get_set5_id, get_set6_id

def create_pattern_vector(values, end_index, length=300):
    """
    Creates a single pattern vector for the window ending at end_index.
    Vector = [Normalized_Values... Set1_ids... Set2_ids... Set3_ids...]
    """
    if end_index < length - 1:
        return None
        
    window = values[end_index - length + 1 : end_index + 1]
    
    # 1. Numeric Values (Normalized to be roughly 0-1 range for k-NN)
    # Assuming max multiplier rarely exceeds 100, we divide by 100.
    # We clip at 100 to avoid outliers distorting distance.
    norm_window = np.clip(window, 0, 100) / 100.0
    
    # 2. Categorical IDs
    s1 = [get_set1_id(v) for v in window]
    s2 = [get_set2_id(v) for v in window]
    s3 = [get_set3_id(v) for v in window]
    s4 = [get_set4_id(v) for v in window]
    s5 = [get_set5_id(v) for v in window]
    s6 = [get_set6_id(v) for v in window]
    
    # 3. Psychological Features (Scalar)
    # We need to calculate them on the fly for the pattern
    # RTP Balance (Approximate for the window)
    rtp_balance = np.sum(window) - (len(window) * 0.97)
    
    # Shockwave (Big X)
    # Check if there is a big X in the window
    has_big_x = 1 if np.max(window) >= 10.0 else 0
    
    # Streak Analysis for Pattern
    # Simplified check for the window: Is there a long streak INSIDE this window?
    # We can check the max streak length in this window.
    max_red_streak = 0
    max_green_streak = 0
    curr_red = 0
    curr_green = 0
    
    for v in window:
        if v < 1.5:
            curr_red += 1
            curr_green = 0
            max_red_streak = max(max_red_streak, curr_red)
        else:
            curr_green += 1
            curr_red = 0
            max_green_streak = max(max_green_streak, curr_green)
            
    has_long_red = 1 if max_red_streak >= 8 else 0
    has_long_green = 1 if max_green_streak >= 8 else 0

    # Concatenate all
    # We add scalars at the end
    return np.concatenate([
        norm_window, 
        np.array(s1), np.array(s2), np.array(s3),
        np.array(s4), np.array(s5), np.array(s6),
        np.array([rtp_balance / 100.0]), # Normalize RTP roughly
        np.array([has_big_x]),
        np.array([has_long_red]),
        np.array([has_long_green])
    ])

def build_memory(values, start_index=300):
    """
    Builds the memory bank for Model B.
    
    Returns:
        memory_patterns: Matrix of patterns
        memory_targets: Dictionary or array of targets corresponding to patterns
    """
    patterns = []
    targets = [] # Stores dict of {next_val, p15, p3}
    
    for i in range(start_index, len(values) - 1):
        pat = create_pattern_vector(values, i, length=300)
        if pat is not None:
            patterns.append(pat)
            
            next_val = values[i+1]
            targets.append({
                'val': next_val,
                'p15': 1 if next_val >= 1.5 else 0,
                'p3': 1 if next_val >= 3.0 else 0
            })
            
    return np.array(patterns), targets

def train_model_b(patterns):
    """
    Trains the NearestNeighbors model.
    """
    # Using 'brute' force algorithm to ensure exact matches and utilize computation power
    nbrs = NearestNeighbors(n_neighbors=200, algorithm='brute', metric='manhattan')
    nbrs.fit(patterns)
    return nbrs

def save_memory(nbrs, patterns, targets, output_dir='.'):
    """
    Saves the k-NN model and the data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    joblib.dump({
        'nbrs': nbrs,
        'patterns': patterns,
        'targets': targets
    }, os.path.join(output_dir, 'modelB_memory'))
    print(f"Model B memory saved to {output_dir}")

def load_memory(model_dir='.'):
    """
    Loads Model B memory.
    """
    data = joblib.load(os.path.join(model_dir, 'modelB_memory'))
    return data['nbrs'], data['patterns'], data['targets']

def predict_model_b(nbrs, memory_targets, current_pattern):
    """
    Predicts using k-NN.
    """
    # Ensure 2D array
    if len(current_pattern.shape) == 1:
        current_pattern = current_pattern.reshape(1, -1)
        
    distances, indices = nbrs.kneighbors(current_pattern)
    
    # Aggregate targets of neighbors
    neighbor_targets = [memory_targets[i] for i in indices[0]]
    
    avg_p15 = np.mean([t['p15'] for t in neighbor_targets])
    avg_p3 = np.mean([t['p3'] for t in neighbor_targets])
    avg_val = np.mean([t['val'] for t in neighbor_targets])
    
    return avg_p15, avg_p3, avg_val

def evaluate_model_b(nbrs, memory_targets, test_values, start_index=0):
    """
    Evaluates Model B on a test set.
    """
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    y_true_p15 = []
    y_pred_p15 = []
    y_true_p3 = []
    y_pred_p3 = []
    y_true_x = []
    y_pred_x = []
    
    print(f"Evaluating Model B on {len(test_values) - start_index} samples...")
    
    for i in range(start_index, len(test_values) - 1):
        pat = create_pattern_vector(test_values, i)
        if pat is not None:
            # Predict
            p15, p3, px = predict_model_b(nbrs, memory_targets, pat)
            
            # True values
            true_val = test_values[i+1]
            
            y_true_x.append(true_val)
            y_pred_x.append(px)
            
            y_true_p15.append(1 if true_val >= 1.5 else 0)
            y_pred_p15.append(1 if p15 >= 0.5 else 0) # Threshold 0.5 for accuracy calc
            
            y_true_p3.append(1 if true_val >= 3.0 else 0)
            y_pred_p3.append(1 if p3 >= 0.5 else 0)

    print(f"Model B Accuracy (P1.5): {accuracy_score(y_true_p15, y_pred_p15):.4f}")
    print(f"Model B Accuracy (P3.0): {accuracy_score(y_true_p3, y_pred_p3):.4f}")
    print(f"Model B MSE (X): {mean_squared_error(y_true_x, y_pred_x):.4f}")
