import streamlit as st
from UI import product_ui, model_ui
from pathlib import Path
import pandas as pd
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
from model.predict_model import ModelPredictor

st.set_page_config(
    page_title="Dream11 AI Suite",
    page_icon="🏏",
    layout="wide"
)

# --- Resource Loading (cached for performance) ---
@st.cache_resource
def load_predictor(model_name="lightgbm", gender="male"):
    """Load and cache a specific model predictor with gender support."""
    model_path = f'model_artifacts/{model_name}_{gender}_model.joblib'
    try:
        return ModelPredictor(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        st.info(f"Run: python model/train_model.py --model_type {model_name} --gender {gender}")
        return None

@st.cache_data
def load_data_and_roles():
    """Load all necessary data once and cache it - CSV format."""
    try:
        data_df = pd.read_csv('data/processed/final_model_data.csv', parse_dates=['date'])
        
        # Filter for T20 only
        if 'match_type' in data_df.columns:
            data_df = data_df[data_df['match_type'] == 'T20'].copy()
        
        roles_df = pd.read_csv('data/processed/player_roles.csv')
        
        return data_df, roles_df
    except FileNotFoundError as e:
        st.error(f"Failed to load data files: {e}")
        st.info("Please run the data pipeline first:")
        st.code("""
python data_processing/data_preprocessing.py
python data_processing/feature_engineering.py
python data_processing/generate_roles.py
        """)
        return None, None

# --- Main App Logic ---
st.sidebar.title("⚙️ Configuration")

# Gender Selector 
gender_choice = st.sidebar.selectbox(
    "Cricket Type",
    ("male", "female"),
    help="Select male or female cricket"
)

# Model Selector
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ("lightgbm", "xgboost"),
    help="Select prediction model"
)

# Load resources with gender parameter
predictor = load_predictor(model_choice, gender_choice)
data_df, roles_df = load_data_and_roles()

# Filter data by gender if loaded successfully
if data_df is not None and 'gender' in data_df.columns:
    data_df = data_df[data_df['gender'] == gender_choice].copy()
    st.sidebar.success(f"✅ Loaded {len(data_df):,} {gender_choice} T20 records")

if all(v is not None for v in [predictor, data_df, roles_df]):
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Go to", 
        ["Team Recommender", "Model Evaluator"],
        help="Choose between building a fantasy team or evaluating model performance"
    )
    
    # Display model info
    if predictor.metrics:
        st.sidebar.info(f"""
**Model Performance**
- RMSE: {predictor.metrics.get('rmse', 0):.2f}
- MAE: {predictor.metrics.get('mae', 0):.2f}
- R²: {predictor.metrics.get('r2', 0):.4f}
        """)
    
    if page == "Team Recommender":
        product_ui.show_page(predictor, data_df, roles_df, gender_choice)
    elif page == "Model Evaluator":
        model_ui.show_page(data_df, roles_df, model_choice, gender_choice)
else:
    st.error("❌ Unable to load required resources. Please check the error messages above.")
