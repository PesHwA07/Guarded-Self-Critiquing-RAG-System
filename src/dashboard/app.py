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
