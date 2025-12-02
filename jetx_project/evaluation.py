import numpy as np
from sklearn.metrics import confusion_matrix

def detailed_evaluation(y_true, y_prob, threshold=0.5, model_name="Model", target_name="1.5x"):
    """
    Prints detailed evaluation metrics including Confusion Matrix and Expected Value.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
        elif "3.0" in target_name:
            profit = (tp * 2.0) - (fp * 1.0)
        else:
            profit = 0
            
        print(f"Tahmini Birim Kâr/Zarar: {profit:.1f} birim")
    else:
        print("Model bu eşikte hiç bahis yapmadı.")
        
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
