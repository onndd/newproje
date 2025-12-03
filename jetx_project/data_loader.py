
import sqlite3
import pandas as pd
import os
from .config import DB_PATH

def load_data(db_path=DB_PATH):
    """
    Connects to the SQLite database and loads the jetx_results table.
    Returns a pandas DataFrame with 'id' and 'value' columns, sorted by id.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at: {db_path}")

    with sqlite3.connect(db_path) as conn:
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
