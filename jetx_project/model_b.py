
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
    # Strict Length Check
    # We need exactly 'length' items ending at end_index.
    # Start index would be: end_index - length + 1
    start_index = end_index - length + 1
    
    if start_index < 0:
        return None
        
    window = values[start_index : end_index + 1]
    
    if len(window) != length:
        return None
    
    # 1. Numeric Values (Logarithmic Scaling)
    # We use log1p to handle large multipliers (e.g. 5000x) without them dominating,
    # while preserving the relative magnitude differences.
    # Normalized by log1p(1000) approx 6.9 to keep range roughly 0-1.
    norm_window = np.log1p(window) / np.log1p(1000.0)
    
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
    
    # Vectorized Streak Analysis
    # Find runs of Red (<1.5) and Green (>=1.5)
    is_red = (window < 1.5)
    
    # Find changes (where value != prev_value)
    # We treat the first element as a change to start the first run
    changes = np.concatenate(([True], is_red[1:] != is_red[:-1]))
    
    # Cumulative sum to assign IDs to runs
    run_ids = np.cumsum(changes)
    
    # Count lengths of each run
    # bincount works on non-negative integers
    # run_ids starts at 1
    run_lengths = np.bincount(run_ids)
    # run_lengths[0] is count of 0s (unused), run_lengths[k] is length of run k
    
    # Get the value (Red or Green) for each run
    # We can take the value at the start index of each run
    run_starts = np.where(changes)[0]
    run_values = is_red[run_starts] # True if Red, False if Green
    
    # Now find max streak for Red and Green
    # run_lengths has an extra 0 at index 0, so we align
    # run_ids goes from 1 to N. run_lengths size is N+1.
    # valid run lengths are run_lengths[1:]
    valid_lengths = run_lengths[1:]
    
    if len(valid_lengths) > 0:
        # Mask for Red runs
        red_runs = valid_lengths[run_values]
        max_red_streak = np.max(red_runs) if len(red_runs) > 0 else 0
        
        # Mask for Green runs
        green_runs = valid_lengths[~run_values]
        max_green_streak = np.max(green_runs) if len(green_runs) > 0 else 0
    else:
        max_red_streak = 0
        max_green_streak = 0
            
    has_long_red = 1 if max_red_streak >= 8 else 0
    has_long_green = 1 if max_green_streak >= 8 else 0

    # Concatenate all
    # We add scalars at the end
    # Normalize Category IDs (approx max 5) to balance weight with numeric values (0-1)
    return np.concatenate([
        norm_window, 
        np.array(s1) / 5.0, np.array(s2) / 5.0, np.array(s3) / 5.0,
        np.array(s4) / 5.0, np.array(s5) / 5.0, np.array(s6) / 5.0,
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

def train_model_b(patterns, n_components=50):
    """
    Trains the NearestNeighbors model with PCA dimensionality reduction.
    """
    from sklearn.decomposition import PCA
    
    # 1. PCA Reduction
    # Reduce dimensions to save memory and speed up query
    # Check if we have enough samples for PCA
    n_samples = len(patterns)
    n_comp = min(n_components, n_samples, patterns.shape[1])
    
    pca = PCA(n_components=n_comp)
    patterns_reduced = pca.fit_transform(patterns)
    
    # 2. Train k-NN
    # Using 'auto' allows scikit-learn to choose the best algorithm (BallTree, KDTree, or Brute)
    # based on the data structure, which is often faster and more memory efficient than forcing 'brute'.
    nbrs = NearestNeighbors(n_neighbors=200, algorithm='auto', metric='manhattan')
    nbrs.fit(patterns_reduced)
    
    return nbrs, pca

def save_memory(nbrs, pca, patterns, targets, output_dir='.'):
    """
    Saves the k-NN model, PCA, and the data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    joblib.dump({
        'nbrs': nbrs,
        'pca': pca,
        # 'patterns': patterns,  <-- REMOVED to save space (200MB+ -> <10MB)
        # We don't need raw patterns for inference, only the tree (nbrs) and PCA.
        'targets': targets
    }, os.path.join(output_dir, 'modelB_memory'))
    print(f"Model B memory saved to {output_dir} (Optimized: No raw patterns)")

def load_memory(model_dir='.'):
    """
    Loads Model B memory.
    """
    data = joblib.load(os.path.join(model_dir, 'modelB_memory'))
    # Return None for patterns as they are not saved anymore
    return data['nbrs'], data.get('pca'), data.get('patterns', None), data['targets']

def predict_model_b(nbrs, pca, memory_targets, current_pattern):
    """
    Predicts using k-NN with PCA.
    """
    # Ensure 2D array
    if len(current_pattern.shape) == 1:
        current_pattern = current_pattern.reshape(1, -1)
        
    # Transform with PCA if available
    if pca is not None:
        current_pattern_reduced = pca.transform(current_pattern)
    else:
        current_pattern_reduced = current_pattern
        
    distances, indices = nbrs.kneighbors(current_pattern_reduced)
    
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
