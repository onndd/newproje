import numpy as np
from sklearn.metrics import confusion_matrix

def detailed_evaluation(y_true, y_prob, threshold=0.5, model_name="Model", target_name="1.5x"):
    """
    Evaluates model performance based on 4 specific scenarios:
    1. FN: Model Under / Real Over (Missed Opportunity)
    2. TP: Model Over / Real Over (Win)
    3. TN: Model Under / Real Under (Safe)
    4. FP: Model Over / Real Under (Loss) - CRITICAL
    """
    # Convert probabilities to binary predictions based on threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate Confusion Matrix
    # tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print(f"\n--- {model_name} - {target_name} Detaylı Analiz (Eşik: {threshold}) ---")
    print(f"1. Model ALT dedi, Gerçekte ÜST geldi (Kaçan Fırsat - FN): {fn}")
    print(f"2. Model ÜST dedi, Gerçekte ÜST geldi (KAZANÇ - TP): {tp}")
    print(f"3. Model ALT dedi, Gerçekte ALT geldi (Güvenli - TN): {tn}")
    print(f"4. Model ÜST dedi, Gerçekte ALT geldi (KAYIP - FP): {fp}  <-- KRİTİK!")
    
    total_bets = tp + fp
    if total_bets > 0:
        win_rate = tp / total_bets * 100
        print(f"Bahis Sayısı: {total_bets}")
        print(f"Kazanma Oranı (Win Rate): %{win_rate:.2f}")
        
        # Simple Profit Calculation (Assuming flat bet)
        # 1.5x: Win = 0.5 unit, Loss = 1 unit
        # 3.0x: Win = 2.0 units, Loss = 1 unit
        if "1.5" in target_name:
            profit = (tp * 0.5) - (fp * 1.0)
        elif "3.0" in target_name:
            profit = (tp * 2.0) - (fp * 1.0)
        else:
            profit = 0
            
        print(f"Tahmini Birim Kâr/Zarar: {profit:.1f} birim")
    else:
        print("Model bu eşikte hiç bahis yapmadı.")
        
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
