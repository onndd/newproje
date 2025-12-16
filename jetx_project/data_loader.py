
import sqlite3
import pandas as pd
import os
from .config import DB_PATH

def load_data(db_path=DB_PATH, limit=None):
    """
    Loads data from the SQLite database.
    If limit is provided, loads only the last N records (efficiently).
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at: {db_path}")

    # Fix: Add timeout to prevent locking issues
    with sqlite3.connect(db_path, timeout=30) as conn:
        # Fix: Check if table exists first
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jetx_results'")
        if cursor.fetchone() is None:
            # Table doesn't exist, return empty DataFrame
            return pd.DataFrame(columns=['id', 'value'])

        if limit:
            # Use rowid to sort by insertion order efficiently
            # Assuming 'id' and 'value' are the columns of interest,
            # and 'rowid' implies the order of insertion.
            # We fetch all columns for simplicity, then filter if needed.
            # Use parameterized query to prevent SQL Injection
            # Ensure limit is an integer
            limit = int(limit)
            query = "SELECT id, value FROM jetx_results ORDER BY rowid DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(limit,))
            # Reverse to chronological order (oldest to newest)
            df = df.iloc[::-1].reset_index(drop=True)
        else:
            query = "SELECT id, value FROM jetx_results ORDER BY id ASC"
            df = pd.read_sql_query(query, conn)
    
    # Ensure 'value' is numeric (float), handle potential string issues
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    
    return df

def split_train_test(df, train_ratio=0.8):
    """
    Splits the data into training and testing sets based on time order.
    NO SHUFFLING.
    
    Args:
        df: pandas DataFrame containing the data
        train_ratio: float, percentage of data to use for training (default 0.8)
        
    Returns:
        train_df, test_df
    """
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    return train_df, test_df

def get_values_array(df):
    """
    Returns the 'value' column as a numpy array.
    """
    return df['value'].values

def add_target_columns(df):
    """
    Adds target columns for training:
    - target_p15: 1 if value >= 1.50
    - target_p3: 1 if value >= 3.00
    - target_crash: 1 if value <= 1.20 (The Danger Zone)
    """
    df = df.copy()
    vals = df['value']
    
    df['target_p15'] = (vals >= 1.50).astype(int)
    df['target_p3'] = (vals >= 3.00).astype(int)
    
    # CRASH DETECTOR TARGET
    # We define 'Crash' as an immediate bust <= 1.20
    # This is what the Guard Model will try to predict.
    df['target_crash'] = (vals <= 1.20).astype(int)
    
    return df
