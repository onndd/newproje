import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def detailed_evaluation(y_true, y_pred_proba, threshold=0.75):
    """
    Prints detailed evaluation metrics including Confusion Matrix, Precision, Recall, F1, ROC-AUC, and Profit/Loss.
    """
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    print(f"--- Evaluation (Threshold: {threshold}) ---")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
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
