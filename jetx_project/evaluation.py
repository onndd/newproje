import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def detailed_evaluation(y_true, y_pred_proba, model_name="Model", threshold=0.70):
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
            
        print(f"False Alarms (FP - Money Lost): {fp}")
        print(f"Missed Opportunities (FN - Profit Lost): {fn}")

        # Confidence Histogram
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Plot Confidence Histogram
            plt.figure(figsize=(10, 4))
            sns.histplot(y_pred_proba, bins=20, kde=True, color='skyblue', label='Pred Probabilities')
            plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
            plt.title(f'{model_name} Confidence Distribution')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Count')
            plt.legend()
            plt.show()
            print("Confidence Histogram displayed.")
            
            # --- NEW: Bankroll Simulation Graph ---
            if tp + fp > 0: # Only plot if we made bets
                # Determine odds based on model name
                win_profit = 0.5 if ("1.5" in model_name or "P1.5" in model_name) else 2.0 if ("3.0" in model_name or "P3.0" in model_name) else 0.0
                
                # Calculate profit per round
                # If we bet (pred=1): 
                #   Win (true=1) -> +win_profit
                #   Lose (true=0) -> -1.0
                # If we don't bet (pred=0) -> 0.0
                
                rewards = np.zeros_like(y_true, dtype=float)
                # Map conditions
                # Win: Pred=1 & True=1
                mask_win = (y_pred == 1) & (y_true == 1)
                # Loss: Pred=1 & True=0
                mask_loss = (y_pred == 1) & (y_true == 0)
                
                rewards[mask_win] = win_profit
                rewards[mask_loss] = -1.0
                
                cumulative_profit = np.cumsum(rewards)
                
                plt.figure(figsize=(12, 5))
                plt.plot(cumulative_profit, color='green' if cumulative_profit[-1] >= 0 else 'red', linewidth=2)
                plt.axhline(0, color='black', linestyle='--', linewidth=1)
                plt.title(f'{model_name} - Bankroll Simulation (Test Run)')
                plt.xlabel('Number of Hands (Test Set)')
                plt.ylabel('Cumulative Profit (Units)')
                plt.grid(True, alpha=0.3)
                
                # Add stats annotation
                final_profit = cumulative_profit[-1]
                max_dd = np.min(cumulative_profit - np.maximum.accumulate(cumulative_profit))
                stats_text = f"Final: {final_profit:.1f}u\nMax DD: {max_dd:.1f}u"
                plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.show()
                print("Bankroll Simulation Graph displayed.")
            # --------------------------------------
        except Exception as e:
            print(f"Could not plot histogram: {e}")
            
        return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "Precision": precision, "Recall": recall}
    else:
        print("Confusion Matrix shape is not (2,2). Skipping detailed metrics.")
        return {}
