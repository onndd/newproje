
import pytest
import sqlite3
import os
import pandas as pd
from jetx_project.config import DB_PATH
from jetx_project.data_loader import load_data

def test_database_exists():
    """Test if the database file exists."""
    assert os.path.exists(DB_PATH), f"Database file not found at {DB_PATH}"

def test_database_connection():
    """Test if we can connect to the database."""
    conn = sqlite3.connect(DB_PATH)
    assert conn is not None
    conn.close()

def test_jetx_results_table():
    """Test if jetx_results table exists and has data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jetx_results';")
    table = cursor.fetchone()
    assert table is not None, "jetx_results table does not exist"
    
    # Check if we can read data
    df = pd.read_sql("SELECT * FROM jetx_results LIMIT 5", conn)
    assert not df.empty, "jetx_results table is empty"
    assert 'value' in df.columns, "value column missing in jetx_results"
    conn.close()
