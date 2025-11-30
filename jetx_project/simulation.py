
import pandas as pd
import numpy as np

class Bankroll:
    def __init__(self, initial_balance=1000):
        self.balance = initial_balance
        self.history = [initial_balance]
        self.max_balance = initial_balance
        self.max_drawdown = 0
        self.current_losing_streak = 0
        self.max_losing_streak = 0

    def update(self, profit):
        self.balance += profit
        self.history.append(self.balance)
        
        # Drawdown calculation
        if self.balance > self.max_balance:
            self.max_balance = self.balance
        
        drawdown = self.max_balance - self.balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            
        # Streak calculation
        if profit < 0:
            self.current_losing_streak += 1
            if self.current_losing_streak > self.max_losing_streak:
                self.max_losing_streak = self.current_losing_streak
        else:
            self.current_losing_streak = 0

    def get_stats(self):
        return {
            "Final Balance": self.balance,
            "Max Drawdown": self.max_drawdown,
            "Max Losing Streak": self.max_losing_streak,
            "Profit": self.balance - self.history[0]
        }

def run_simulation(predictions_df, model_name="Model A"):
    """
    Runs the simulation for 3 strategies based on predictions.
    
    predictions_df columns expected:
    - 'true_val': Actual X value
    - 'p_1_5': Probability of X >= 1.5
    - 'p_3': Probability of X >= 3.0
    - 'pred_x': Predicted X value
    """
    
    kasa1 = Bankroll()
    kasa2 = Bankroll()
    kasa3 = Bankroll()
    
    for _, row in predictions_df.iterrows():
        true_val = row['true_val']
        p_1_5 = row['p_1_5']
        p_3 = row['p_3']
        pred_x = row['pred_x']
        
        # --- Kasa 1: 1.5x, %70 confidence ---
        if p_1_5 >= 0.70:
            bet = 10
            target = 1.5
            if true_val >= target:
                profit = bet * (target - 1)
            else:
                profit = -bet
            kasa1.update(profit)
        else:
            kasa1.update(0) # No bet
            
        # --- Kasa 2: 1.5x, %80 confidence ---
        if p_1_5 >= 0.80:
            bet = 20
            target = 1.5
            if true_val >= target:
                profit = bet * (target - 1)
            else:
                profit = -bet
            kasa2.update(profit)
        else:
            kasa2.update(0)
            
        # --- Kasa 3: 3x focus, %70 confidence ---
        if p_3 >= 0.70:
            bet = 10
            # Target exit: max(1.5, 0.8 * x_pred)
            target = max(1.5, 0.8 * pred_x)
            
            if true_val >= target:
                profit = bet * (target - 1)
            else:
                profit = -bet
            kasa3.update(profit)
        else:
            kasa3.update(0)
            
    # Report
    print(f"--- Simulation Results for {model_name} ---")
    print(f"Kasa 1 (1.5x @ 70%): {kasa1.get_stats()}")
    print(f"Kasa 2 (1.5x @ 80%): {kasa2.get_stats()}")
    print(f"Kasa 3 (Dynamic @ 70%): {kasa3.get_stats()}")
    print("-" * 30)
    
    return {
        "Kasa1": kasa1,
        "Kasa2": kasa2,
        "Kasa3": kasa3
    }
