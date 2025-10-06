import streamlit as st
from UI import product_ui, model_ui
from pathlib import Path
import pandas as pd
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
from model.predict_model import ModelPredictor

# --- Page Configuration ---
st.set_page_config(
    page_title="Dream11 AI Suite",
    page_icon="🏏",
    layout="wide"
)

# --- Resource Loading (cached for performance) ---
@st.cache_resource
def load_predictor(model_name="lightgbm"):
    """Load and cache a specific model predictor."""
    model_path = f'model_artifacts/{model_name}_model.joblib'
    try:
        return ModelPredictor(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        return None

@st.cache_data
def load_data_and_roles():
    """Load all necessary CSV data once and cache it."""
    try:
        # CHANGED: Read from CSV and parse the date column
        data_df = pd.read_csv('data/processed/final_model_data.csv', parse_dates=['date'])
        roles_df = pd.read_csv('data/processed/player_roles.csv')
        return data_df, roles_df
    except FileNotFoundError as e:
        st.error(f"Failed to load data files: {e}. Please run the data pipeline first.")
        return None, None

# --- Main App Logic ---
st.sidebar.title("Configuration")

model_choice = st.sidebar.selectbox(
    "Choose a Model",
    ("lightgbm", "xgboost")
)

predictor = load_predictor(model_choice)
data_df, roles_df = load_data_and_roles()

if all(v is not None for v in [predictor, data_df, roles_df]):
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Team Recommender", "Model Evaluator"])
    if page == "Team Recommender":
        product_ui.show_page(predictor, data_df, roles_df)
    elif page == "Model Evaluator":
        model_ui.show_page(data_df, roles_df, model_choice)