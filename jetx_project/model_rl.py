
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class JetXEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    "The Strategist" learns to bet based on Meta-Learner predictions and market state.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=1000.0):
        super(JetXEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        
        # Actions: 0=Pass, 1=Bet 1.5x, 2=Bet 2.0x
        self.action_space = spaces.Discrete(3)
        
        # Observation Space:
        # 1. Meta Prob 1.5x (0-1)
        # 2. Meta Prob 3.0x (0-1) (or similar high risk prob)
        # 3. HMM State (0, 1, 2) -> Normalized to 0-1? Or just raw.
        # 4. Balance (Normalized by initial)
        # 5. Trend (Last 10 games avg) -> Normalized
        
        # We define bounds roughly.
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(5,), 
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = 10 # Start after some history
        
        return self._next_observation(), {}

    def _next_observation(self):
        # Get data for current step
        row = self.df.iloc[self.current_step]
        
        # Trend: Last 10 games average value
        # We need access to previous rows
        past_10 = self.df.iloc[self.current_step-10 : self.current_step]['value']
        trend = past_10.mean()
        
        # Log-Scale Trend (Fix Scaling Mismatch)
        trend_log = np.log1p(trend)
        
        obs = np.array([
            row['prob_1.5'],
            row.get('prob_3.0', 0.0), # Default to 0 if not present
            row['hmm_state'],
            self.balance / self.initial_balance,
            trend_log # Use log-scaled trend
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        # Execute action
        row = self.df.iloc[self.current_step]
        true_val = row['value']
        
        reward = 0
        bet_amount = 10.0 # Fixed bet for simplicity, or dynamic?
        # RL Agent could output continuous action for bet amount, but let's stick to Discrete for now.
        
        if action == 0: # PASS
            # Small penalty for passing good opportunities? 
            # Or 0 reward.
            reward = 0
            
        elif action == 1: # BET 1.5x
            target = 1.5
            if true_val >= target:
                profit = bet_amount * (target - 1)
                self.balance += profit
                reward = profit
            else:
                loss = bet_amount
                self.balance -= loss
                reward = -loss * 1.5 # Higher penalty for losing
                
        elif action == 2: # BET 2.0x
            target = 2.0
            if true_val >= target:
                profit = bet_amount * (target - 1)
                self.balance += profit
                reward = profit * 1.2 # Bonus for hitting high risk
            else:
                loss = bet_amount
                self.balance -= loss
                reward = -loss * 1.5
        
        # Check bankruptcy
        terminated = False
        if self.balance <= 0:
            terminated = True
            reward = -1000 # Big penalty
            
        # Move to next step
        self.current_step += 1
        truncated = False
        if self.current_step >= len(self.df) - 1:
            terminated = True
            
        return self._next_observation(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}")

def train_rl_agent(df_with_probs, output_dir='.'):
    """
    Trains PPO agent.
    df_with_probs must contain: 'value', 'prob_1.5', 'prob_3.0', 'hmm_state'
    """
    print("Training RL Agent (The Strategist)...")
    
    # Create Env
    env = JetXEnv(df_with_probs)
    
    # Initialize Agent
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
    
    # Train
    model.learn(total_timesteps=50000) # Adjust as needed
    
    return model

def predict_action(model, prob_15, prob_30, hmm_state, balance, recent_trend, initial_balance=1000.0):
    """
    Predicts action for a single state.
    """
    # Log-Scale Trend
    trend_log = np.log1p(recent_trend)
    
    # Construct observation
    obs = np.array([
        prob_15,
        prob_30,
        hmm_state,
        balance / initial_balance, # Dynamic Normalization
        trend_log
    ], dtype=np.float32)
    
    action, _ = model.predict(obs, deterministic=True)
    return int(action)

def save_rl_agent(model, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(os.path.join(output_dir, "model_rl_agent"))
    print(f"RL Agent saved to {output_dir}")

def load_rl_agent(model_dir='.'):
    path = os.path.join(model_dir, "model_rl_agent.zip")
    if not os.path.exists(path):
        return None
    return PPO.load(path)
