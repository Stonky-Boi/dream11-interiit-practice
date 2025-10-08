"""
Main Application - Dream11 Inter-IIT Project
Entry point for both Product UI and Model UI
"""

import streamlit as st
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent))

def main():
    st.set_page_config(
        page_title="Dream11 Team Builder",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    st.sidebar.title("🏏 Dream11 AI")
    st.sidebar.markdown("### Navigation")
    
    page = st.sidebar.radio(
        "Select Interface",
        ["🏏 Product UI - Team Builder", "🔬 Model UI - Evaluation"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    **Dream11 Inter-IIT Tech Meet 13.0**
    
    AI-powered fantasy cricket team builder using:
    - **60+ Features** (Silver Medal Team approach)
    - **Ensemble ML** (XGBoost + LightGBM + CatBoost)
    - **7,000+ Matches** from Cricsheet
    - **Career Aggregate Stats** integration
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    
    # Check if models exist
    model_path = Path('model_artifacts/ProductUIModel_metadata.json')
    if model_path.exists():
        with open(model_path, 'r') as f:
            metadata = json.load(f)
        st.sidebar.success(f"✅ Models loaded ({metadata.get('num_features', 0)} features)")
        st.sidebar.metric("Model MAE", f"{metadata.get('ensemble_mae', 0):.2f}")
    else:
        st.sidebar.warning("⚠️ No trained model found")
        st.sidebar.info("Train a model using Model UI")
    
    # Check if data exists
    data_path = Path('data/processed/training_data_2024-06-30.csv')
    if data_path.exists():
        st.sidebar.success("✅ Training data ready")
    else:
        st.sidebar.warning("⚠️ No training data found")
        st.sidebar.info("Run data processing scripts")
    
    # Route to selected page
    if page == "🏏 Product UI - Team Builder":
        from UI.product_ui import ProductUI
        ui = ProductUI()
        ui.run()
    
    elif page == "🔬 Model UI - Evaluation":
        from UI.model_ui import ModelUI
        ui = ModelUI()
        ui.run()


if __name__ == '__main__':
    main()