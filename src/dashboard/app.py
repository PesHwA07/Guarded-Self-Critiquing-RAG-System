import sqlite3
import pandas as pd
import streamlit as st
import os
import sys
from pathlib import Path

# --- Configuration ---
# Ensure we are pointing to the correct DB path relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "eval_history.db"

# Ensure src/ is in the python path so we can import rag modules
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from rag.graph import run_query

st.set_page_config(
    page_title="Guarded RAG Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# --- Data Loading ---
@st.cache_data(ttl=60) # Cache the data for 60 seconds
def load_eval_data(db_path: Path) -> pd.DataFrame:
    """Load evaluation history from SQLite into a Pandas DataFrame."""
    if not db_path.exists():
        return pd.DataFrame()
        
    try:
        conn = sqlite3.connect(db_path)
        # We order by timestamp descending so the newest runs are first
        query = "SELECT * FROM eval_runs ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert timestamp strings to datetime objects
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return df
    except Exception as e:
        st.error(f"Error loading evaluation data: {e}")
        return pd.DataFrame()

# --- Layout ---
st.title("🛡️ Guarded RAG System Dashboard")
st.markdown("""
Welcome to the Guarded RAG System Dashboard. 
This dashboard tracks the evaluation metrics and pipeline health of our self-critiquing AI system.