
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
    kasa4 = Bankroll()
    
    # Standardize column names
    df = predictions_df.copy()
    
    # Map 'value' to 'true_val' if needed
    if 'true_val' not in df.columns and 'value' in df.columns:
        df['true_val'] = df['value']
        
    # Map 'prob_1.5' to 'p_1_5'
    if 'p_1_5' not in df.columns and 'prob_1.5' in df.columns:
        df['p_1_5'] = df['prob_1.5']
        
    # Map 'prob_3.0' to 'p_3'
    if 'p_3' not in df.columns and 'prob_3.0' in df.columns:
        df['p_3'] = df['prob_3.0']
        
    # Map 'pred_value' to 'pred_x'
    if 'pred_x' not in df.columns and 'pred_value' in df.columns:
        df['pred_x'] = df['pred_value']
        
    # Check for missing columns
    required_cols = ['true_val', 'p_1_5', 'p_3', 'pred_x']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns for simulation: {missing}")
        return None
        
    # Stop-Loss Flags
    kasa1_active = True
    kasa2_active = True
    kasa3_active = True
    kasa4_active = True
    
    # Drawdown Limit (e.g., Stop if we lose 50% of peak bankroll)
    DRAWDOWN_LIMIT = max_drawdown_limit 
    
    for _, row in df.iterrows():
        true_val = row['true_val']
        p_1_5 = row['p_1_5']
        p_3 = row['p_3']
        pred_x = row['pred_x']
        
        # Check Stop-Loss Conditions
        if kasa1.max_drawdown > (kasa1.max_balance * DRAWDOWN_LIMIT): kasa1_active = False
        if kasa2.max_drawdown > (kasa2.max_balance * DRAWDOWN_LIMIT): kasa2_active = False
        if kasa3.max_drawdown > (kasa3.max_balance * DRAWDOWN_LIMIT): kasa3_active = False
        if kasa4.max_drawdown > (kasa4.max_balance * DRAWDOWN_LIMIT): kasa4_active = False
        
        # --- Kasa 1: 1.5x, %75 confidence ---
        if kasa1_active and p_1_5 >= 0.75:
            bet = 10
            target = 1.5
            if true_val > target:
                profit = bet * (target - 1)
            else:
                profit = -bet
            kasa1.update(profit)
        else:
            kasa1.update(0) # No bet
            
        # --- Kasa 2: 1.5x, %85 confidence ---
        if kasa2_active and p_1_5 >= 0.85:
            bet = 20
            target = 1.5
            if true_val > target:
                profit = bet * (target - 1)
            else:
                profit = -bet
            kasa2.update(profit)
        else:
            kasa2.update(0)
            
        # --- Kasa 3: 3x focus, %55 confidence ---
        if kasa3_active and p_3 >= 0.55:
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

        # --- Kasa 4: Smart Kelly (Dynamic Staking) ---
        if kasa4_active and p_1_5 > 0.65:
            target = 1.50
            b = target - 1 # 0.5
            p = p_1_5
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Safety: Cap Kelly at 5% of bankroll
            bet_fraction = min(max(kelly_fraction, 0), 0.05)
            
            if bet_fraction > 0:
                bet = kasa4.balance * bet_fraction
                bet = max(bet, 1.0)
                
                if true_val > target:
                    profit = bet * (target - 1)
                else:
                    profit = -bet
                kasa4.update(profit)
            else:
                kasa4.update(0)
        else:
            kasa4.update(0)
            
    # Report
    print(f"--- Simulation Results for {model_name} ---")
    print(f"Kasa 1 (1.5x @ 75%): {kasa1.get_stats()} {'(STOPPED)' if not kasa1_active else ''}")
    print(f"Kasa 2 (1.5x @ 85%): {kasa2.get_stats()} {'(STOPPED)' if not kasa2_active else ''}")
    print(f"Kasa 3 (Dynamic @ 55%): {kasa3.get_stats()} {'(STOPPED)' if not kasa3_active else ''}")
    print(f"Kasa 4 (Smart Kelly): {kasa4.get_stats()} {'(STOPPED)' if not kasa4_active else ''}")
    print("-" * 30)
    
    return {
        "Kasa1": kasa1,
        "Kasa2": kasa2,
        "Kasa3": kasa3,
        "Kasa4": kasa4
    }
