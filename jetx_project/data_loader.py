
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

    conn = sqlite3.connect(db_path)
    try:
        if limit:
            # Use rowid to sort by insertion order efficiently
            # Assuming 'id' and 'value' are the columns of interest,
            # and 'rowid' implies the order of insertion.
            # We fetch all columns for simplicity, then filter if needed.
            query = f"SELECT id, value FROM jetx_results ORDER BY rowid DESC LIMIT {limit}"
            df = pd.read_sql_query(query, conn)
            # Reverse to chronological order (oldest to newest)
            df = df.iloc[::-1].reset_index(drop=True)
        else:
            query = "SELECT id, value FROM jetx_results ORDER BY id ASC"
            df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    
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
