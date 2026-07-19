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
""")

st.sidebar.header("Dashboard Controls")
st.sidebar.markdown("Use this sidebar to filter or refresh data.")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()

# Load the data
with st.spinner("Loading evaluation history..."):
    df_evals = load_eval_data(DB_PATH)

if df_evals.empty:
    st.info("No evaluation data found. Run an evaluation to populate the database.")
else:
    # --- Quick Stats ---
    st.subheader("Latest Evaluation Run (Full Mode)")
    
    # Get the latest 'full' eval run
    full_runs = df_evals[df_evals['mode'] == 'full']
    if not full_runs.empty:
        latest = full_runs.iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Hallucination Rate", f"{latest['hallucination_rate']*100:.1f}%")
        col2.metric("Avg Relevancy", f"{latest['average_relevancy']:.2f}")
        col3.metric("Latency (p95)", f"{latest['latency_p95_sec']:.2f}s")
        col4.metric("Cost per Query", f"${latest['cost_per_query']:.4f}")
    else:
        st.info("No 'full' mode evaluation runs available yet.")

    st.divider()

    # Create tabs for Analytics, Raw Data, and Debugger
    tab1, tab2, tab3 = st.tabs(["📈 Analytics Trends", "🗄️ Raw Data", "🔍 Query Debugger"])

    with tab1:
        st.subheader("Evaluation Trends Over Time")
        
        # Sort values chronologically for charts
        df_charts = df_evals.sort_values("timestamp").copy()
        df_charts.set_index("timestamp", inplace=True)
        
        # We only want to plot runs that succeeded (total_queries > 0)
        df_charts = df_charts[df_charts["total_queries"] > 0]
        
        if not df_charts.empty:
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("**Hallucination Rate Trend**")
                # Convert to percentage
                st.line_chart(df_charts["hallucination_rate"] * 100, height=250)
                
                st.markdown("**Cost Per Query Trend (USD)**")
                st.area_chart(df_charts["cost_per_query"], height=250)
                
            with chart_col2:
                st.markdown("**Latency (p95) Trend (sec)**")
                st.line_chart(df_charts["latency_p95_sec"], height=250)