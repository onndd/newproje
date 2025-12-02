import numpy as np
from sklearn.metrics import confusion_matrix

def detailed_evaluation(y_true, y_prob, threshold=0.5, model_name="Model", target_name="1.5x"):
    """
    Prints detailed evaluation metrics including Confusion Matrix and Expected Value.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    print(f"\n--- {model_name} Evaluation ({target_name}) ---")
    print(f"Threshold: {threshold}")
    print(f"Confusion Matrix: [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Expected Value Calculation (Approximate)
    # Assuming 1 unit bet.
    # Cost of FP = -1 unit.
    # Profit of TP = (Target - 1) unit.
    
    if tp + fp > 0:
        if "1.5" in target_name:
            profit = (tp * 0.5) - (fp * 1.0)
        elif "3.0" in target_name:
            profit = (tp * 2.0) - (fp * 1.0)
        else:
            profit = 0
            
        print(f"Estimated Profit (Unit): {profit:.1f} units")
        print(f"Win Rate (on bets): {tp / (tp + fp):.2%}")
    else:
        print("Model made NO bets at this threshold.")
        
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "Precision": precision, "Recall": recall}
