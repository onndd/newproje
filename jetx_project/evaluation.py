import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def detailed_evaluation(y_true, y_pred_proba, model_name="Model", threshold=0.75):
    """
    Prints detailed evaluation metrics including Confusion Matrix, Precision, Recall, F1, ROC-AUC, and Profit/Loss.
    """
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    print(f"\n--- {model_name} Evaluation (Threshold: {threshold}) ---")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        
        # ROC AUC
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
            print(f"ROC-AUC: {auc:.4f}")
        except:
            print("ROC-AUC: N/A")

        # Profit Calculation
        if tp + fp > 0:
            if "1.5" in model_name or "P1.5" in model_name:
                profit = (tp * 0.5) - (fp * 1.0)
            elif "3.0" in model_name or "P3.0" in model_name:
                profit = (tp * 2.0) - (fp * 1.0)
            else:
                profit = 0 # Default
                
            print(f"Estimated Profit (Unit): {profit:.1f} units")
            print(f"Win Rate (on bets): {tp / (tp + fp):.2%}")
        else:
            print("Model made NO bets at this threshold.")
            
        return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "Precision": precision, "Recall": recall}
    else:
        print("Confusion Matrix shape is not (2,2). Skipping detailed metrics.")
        return {}
