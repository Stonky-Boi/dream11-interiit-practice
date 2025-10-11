import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
sys.path.append(str(Path(__file__).parent.parent))

from model.train_model import Dream11ModelTrainer
from model.predict_model import Dream11Predictor
from datetime import datetime, date
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ModelUI:
    def __init__(self):
        self.data_dir = Path('data')
        self.processed_dir = self.data_dir / 'processed'
        self.model_artifacts_dir = Path('model_artifacts')
    
    def train_model(self, train_start, train_end, model_name):
        st.info(f"📚 Training model from {train_start} to {train_end}...")
        
        # Load all data
        data_path = self.processed_dir / 'training_data_2024-06-30.csv'
        if not data_path.exists():
            st.error("Training data not found. Please run feature engineering first.")
            return None
        
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert dates to pandas Timestamp
        train_start = pd.to_datetime(train_start)
        train_end = pd.to_datetime(train_end)
        
        # Filter by training range
        train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
        
        st.write(f"Training samples: {len(train_df):,}")
        st.write(f"Date range: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
        st.write(f"Features: {len([c for c in train_df.columns if c not in ['player', 'match_id', 'date', 'venue', 'team', 'opposition', 'fantasy_points']])}")
        
        # Save filtered training data
        train_filename = f"training_data_{train_end.strftime('%Y-%m-%d')}.csv"
        train_df.to_csv(self.processed_dir / train_filename, index=False)
        
        # Train model with progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Create trainer
            trainer = Dream11ModelTrainer(
                data_path=self.processed_dir / train_filename,
                model_artifacts_dir=str(self.model_artifacts_dir)
            )
            
            # Load and prepare
            status_text.text("Loading data...")
            progress_bar.progress(10)
            trainer.load_data(train_end_date=train_end.strftime('%Y-%m-%d'))
            
            status_text.text("Preparing features (60+)...")
            progress_bar.progress(20)
            trainer.prepare_features()
            trainer.split_data()
            
            # Train baseline models
            status_text.text("Training baseline models...")
            progress_bar.progress(30)
            trainer.train_baseline_models()
            
            # Train ensemble models
            status_text.text("Training XGBoost...")
            progress_bar.progress(45)
            trainer.train_xgboost()
            
            status_text.text("Training LightGBM...")
            progress_bar.progress(60)
            trainer.train_lightgbm()
            
            status_text.text("Training CatBoost...")
            progress_bar.progress(75)
            trainer.train_catboost()
            
            status_text.text("Creating ensemble...")
            progress_bar.progress(90)
            trainer.create_ensemble()
            
            status_text.text("Saving models...")
            trainer.save_models(model_name)
            
            progress_bar.progress(100)
            status_text.text("✅ Training complete!")
            
            return trainer
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    def evaluate_model(self, model_name, test_start, test_end):
        st.info(f"🧪 Evaluating on period {test_start} to {test_end}...")
        
        try:
            # Load predictor
            predictor = Dream11Predictor(
                model_dir=str(self.model_artifacts_dir),
                model_name=model_name
            )
            
            # Convert dates to pandas Timestamp
            test_start = pd.to_datetime(test_start)
            test_end = pd.to_datetime(test_end)
            
            # Load test data
            data_path = self.processed_dir / 'training_data_2024-06-30.csv'
            if not data_path.exists():
                st.error("No processed data found. Run feature engineering first.")
                return None
            
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter test period
            test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)]
            
            if len(test_df) == 0:
                st.error(f"❌ No matches found in period {test_start.date()} to {test_end.date()}")
                st.info(f"Available date range: {df['date'].min().date()} to {df['date'].max().date()}")
                return None
            
            st.write(f"✓ Test samples: {len(test_df):,}")
            st.write(f"✓ Unique matches: {test_df['match_id'].nunique():,}")
            
            # Get unique matches
            matches = test_df.groupby(['match_id', 'date']).first().reset_index()
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (_, match) in enumerate(matches.iterrows()):
                progress = (idx + 1) / len(matches)
                progress_bar.progress(progress)
                status_text.text(f"Evaluating match {idx + 1}/{len(matches)}...")
                
                match_data = test_df[test_df['match_id'] == match['match_id']].copy()
                
                if len(match_data) < 11:
                    continue
                
                # Predict
                predictions = predictor.predict(match_data)
                predictions = predictions.sort_values('predicted_fantasy_points', ascending=False)
                
                # Get actual dream team (top 11 by actual fantasy points)
                actual_dream_team = match_data.nlargest(11, 'fantasy_points')
                
                # Get predicted dream team (top 11 by predicted points)
                predicted_dream_team = predictions.nlargest(11, 'predicted_fantasy_points')
                
                # Calculate MAE
                actual_total = actual_dream_team['fantasy_points'].sum()
                predicted_total = predicted_dream_team['predicted_fantasy_points'].sum()
                mae = abs(actual_total - predicted_total)
                
                result = {
                    'Match_ID': match['match_id'],
                    'Match_Date': match['date'].strftime('%Y-%m-%d'),
                    'Team_1': match.get('team', 'Unknown'),
                    'Team_2': match.get('opposition', 'Unknown'),
                    'Venue': match.get('venue', 'Unknown'),
                    'Match_Type': match.get('match_type', 'Unknown'),
                    'Dream_Team_Total_Points': round(actual_total, 2),
                    'Predicted_Team_Total': round(predicted_total, 2),
                    'MAE': round(mae, 2)
                }
                
                # Add predicted team players
                for i in range(min(11, len(predicted_dream_team))):
                    pred_player = predicted_dream_team.iloc[i]
                    result[f'Predicted_Player_{i+1}'] = pred_player['player']
                    result[f'Predicted_Player_{i+1}_Points'] = round(pred_player['predicted_fantasy_points'], 2)
                    result[f'Predicted_Player_{i+1}_Role'] = pred_player.get('role', 'Unknown')
                
                # Add actual dream team players
                for i in range(min(11, len(actual_dream_team))):
                    actual_player = actual_dream_team.iloc[i]
                    result[f'Dream_Team_Player_{i+1}'] = actual_player['player']
                    result[f'Dream_Team_Player_{i+1}_Points'] = round(actual_player['fantasy_points'], 2)
                    result[f'Dream_Team_Player_{i+1}_Role'] = actual_player.get('role', 'Unknown')
                
                results.append(result)
            
            progress_bar.progress(1.0)
            status_text.text("✓ Evaluation complete!")
            
            if len(results) == 0:
                st.error("No valid matches found for evaluation")
                return None
            
            results_df = pd.DataFrame(results)
            return results_df
            
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    def display_model_comparison(self, model_name):
        comparison_path = self.model_artifacts_dir / f"{model_name}_model_comparison.json"
        
        if not comparison_path.exists():
            st.warning("No model comparison data found")
            return
        
        with open(comparison_path, 'r') as f:
            comparison_data = json.load(f)
        
        st.markdown("### 📊 Model Performance Comparison")
        
        # Create comparison dataframe
        models_data = []
        for model_name_key, results in comparison_data['all_models'].items():
            models_data.append({
                'Model': model_name_key.replace('_', ' ').title(),
                'Type': results['type'].title(),
                'Train MAE': results['train_mae'],
                'Val MAE': results['val_mae'],
                'Val RMSE': results['val_rmse'],
                'Val R²': results['val_r2']
            })
        
        df_comparison = pd.DataFrame(models_data).sort_values('Val MAE')
        
        # Display table
        st.dataframe(df_comparison, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # MAE comparison
            import plotly.express as px
            fig_mae = px.bar(
                df_comparison,
                x='Model',
                y='Val MAE',
                color='Type',
                title='Model Comparison - Validation MAE (Lower is Better)',
                labels={'Val MAE': 'Validation MAE'},
                color_discrete_map={'Baseline': '#FF6B6B', 'Ensemble': '#4ECDC4'}
            )
            fig_mae.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_mae, use_container_width=True)
        
        with col2:
            # R² comparison
            fig_r2 = px.bar(
                df_comparison,
                x='Model',
                y='Val R²',
                color='Type',
                title='Model Comparison - R² Score (Higher is Better)',
                labels={'Val R²': 'R² Score'},
                color_discrete_map={'Baseline': '#FF6B6B', 'Ensemble': '#4ECDC4'}
            )
            fig_r2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        # Best model highlight
        best_model = comparison_data['best_model']
        best_mae = df_comparison[df_comparison['Model'] == best_model.replace('_', ' ').title()]['Val MAE'].values[0]
        
        st.success(f"🏆 Best Model: **{best_model.replace('_', ' ').title()}** (Val MAE: {best_mae:.2f})")
    
    def run(self):
        st.title("🔬 Model UI - Performance Analysis")
        st.markdown("---")
        
        # Info box
        st.info("""
        **Competition Requirements:**
        - Training data: Up to 2024-06-30 (already processed)
        - Test data: After 2024-07-01 (will be provided by organizers)
        - Model must be reproducible and generate CSV output
        
        **Options:**
        1. **Train Model**: Train on available data (2024-06-30 cutoff)
        2. **Evaluate on Test Data**: Load and evaluate on out-of-sample test data
        """)
        
        # Select mode
        mode = st.radio(
            "Select Mode",
            ["Train Model Only", "Train & Evaluate on Test Data"],
            horizontal=True
        )
        
        # Training section
        st.header("1️⃣ Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_start = st.date_input(
                "Training Start Date",
                value=datetime(2010, 1, 1),
                min_value=datetime(2000, 1, 1),
                max_value=datetime(2024, 6, 30),
                help="Start date for training data"
            )
        
        with col2:
            train_end = st.date_input(
                "Training End Date",
                value=datetime(2024, 6, 30),
                min_value=datetime(2000, 1, 1),
                max_value=datetime(2024, 6, 30),
                help="MUST be ≤ 2024-06-30 (disqualification if violated)"
            )
        
        model_name = st.text_input(
            "Model Name",
            value=f"Model_{train_end.strftime('%Y%m%d')}",
            help="Give your model a unique name"
        )
        
        # Test data section (only if evaluate mode)
        if mode == "Train & Evaluate on Test Data":
            st.header("2️⃣ Test Data Configuration")
            
            st.markdown("""
            Upload test data CSV with the same format as training data.
            
            **Expected columns:**
            - player, match_id, date, venue, team, opposition, match_type
            - fantasy_points (target)
            - All feature columns (45+ features)
            """)
            
            test_data_source = st.radio(
                "Test Data Source",
                ["Upload CSV File", "Load from data/out_of_sample_data/"]
            )
            
            test_df = None
            
            if test_data_source == "Upload CSV File":
                uploaded_file = st.file_uploader(
                    "Upload Test Data CSV",
                    type=['csv'],
                    help="Test data should have same structure as training data"
                )
                
                if uploaded_file is not None:
                    try:
                        test_df = pd.read_csv(uploaded_file)
                        test_df['date'] = pd.to_datetime(test_df['date'])
                        
                        st.success(f"✅ Loaded test data: {len(test_df):,} records")
                        st.write(f"Date range: {test_df['date'].min().date()} to {test_df['date'].max().date()}")
                        
                        # Validate test data is after cutoff
                        if test_df['date'].min() <= pd.to_datetime('2024-06-30'):
                            st.warning("⚠️ Test data contains dates ≤ 2024-06-30. Should be > 2024-07-01")
                        
                    except Exception as e:
                        st.error(f"Error loading test data: {str(e)}")
            
            else:  # Load from directory
                test_data_dir = Path('data/out_of_sample_data')
                
                if test_data_dir.exists():
                    csv_files = list(test_data_dir.glob('*.csv'))
                    
                    if csv_files:
                        selected_file = st.selectbox(
                            "Select Test Data File",
                            [f.name for f in csv_files]
                        )
                        
                        if selected_file:
                            test_path = test_data_dir / selected_file
                            test_df = pd.read_csv(test_path)
                            test_df['date'] = pd.to_datetime(test_df['date'])
                            
                            st.success(f"✅ Loaded: {selected_file}")
                            st.write(f"Records: {len(test_df):,}")
                            st.write(f"Date range: {test_df['date'].min().date()} to {test_df['date'].max().date()}")
                    else:
                        st.warning("No CSV files found in data/out_of_sample_data/")
                        st.info("Place test data CSV files in data/out_of_sample_data/ directory")
                else:
                    st.warning("Directory data/out_of_sample_data/ does not exist")
                    test_data_dir.mkdir(parents=True, exist_ok=True)
                    st.info("Created directory. Please add test data CSV files.")
        
        # Validation
        cutoff_date = date(2024, 6, 30)
        
        if train_end > cutoff_date:
            st.error("⚠️ DISQUALIFICATION RISK: Training end date must be ≤ 2024-06-30")
        else:
            st.success("✅ Training configuration is valid")
        
        st.markdown("---")
        
        # Action buttons
        if mode == "Train Model Only":
            train_button = st.button(
                "🎯 Train Model",
                type="primary",
                use_container_width=True,
                disabled=(train_end > cutoff_date)
            )
            
            if train_button:
                st.markdown("---")
                st.header("2️⃣ Training Progress")
                
                trainer = self.train_model(train_start, train_end, model_name)
                
                if trainer:
                    st.success(f"✅ Model trained successfully: {model_name}")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Ensemble MAE", f"{trainer.ensemble_mae:.2f}")
                    with col2:
                        st.metric("Ensemble RMSE", f"{trainer.ensemble_rmse:.2f}")
                    with col3:
                        st.metric("Ensemble R²", f"{trainer.ensemble_r2:.4f}")
                    with col4:
                        st.metric("Features", len(trainer.feature_cols))
                    
                    # Display model comparison
                    st.markdown("---")
                    self.display_model_comparison(model_name)
        
        else:  # Train & Evaluate
            eval_button = st.button(
                "📊 Train & Evaluate",
                type="primary",
                use_container_width=True,
                disabled=(train_end > cutoff_date or test_df is None)
            )
            
            if eval_button:
                if test_df is None:
                    st.error("Please load test data first")
                    return
                
                st.markdown("---")
                st.header("3️⃣ Training Progress")
                
                # Train
                trainer = self.train_model(train_start, train_end, model_name)
                
                if trainer:
                    st.success(f"✅ Model trained: {model_name}")
                    
                    # Display model comparison
                    st.markdown("---")
                    self.display_model_comparison(model_name)
                    
                    st.markdown("---")
                    st.header("4️⃣ Evaluation on Test Data")
                    
                    # Evaluate on test data
                    results_df = self.evaluate_on_test_data(model_name, test_df)
                    
                    if results_df is not None and len(results_df) > 0:
                        # Display metrics
                        avg_mae = results_df['MAE'].mean()
                        median_mae = results_df['MAE'].median()
                        min_mae = results_df['MAE'].min()
                        max_mae = results_df['MAE'].max()
                        
                        st.success(f"✅ Evaluation complete!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Matches", len(results_df))
                        with col2:
                            st.metric("Average MAE", f"{avg_mae:.2f}")
                        with col3:
                            st.metric("Median MAE", f"{median_mae:.2f}")
                        with col4:
                            st.metric("Best MAE", f"{min_mae:.2f}")
                        
                        # MAE distribution
                        st.markdown("### MAE Distribution")
                        import plotly.express as px
                        fig = px.histogram(
                            results_df,
                            x='MAE',
                            nbins=30,
                            title='Distribution of Match-level MAE',
                            labels={'MAE': 'Mean Absolute Error'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display results table
                        st.markdown("### Detailed Results")
                        st.dataframe(
                            results_df[[
                                'Match_Date', 'Team_1', 'Team_2', 'Match_Type',
                                'Dream_Team_Total_Points', 'Predicted_Team_Total', 'MAE'
                            ]],
                            use_container_width=True
                        )
                        
                        # Download CSV
                        csv = results_df.to_csv(index=False)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"evaluation_results_{model_name}_{timestamp}.csv"
                        
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name=filename,
                            mime="text/csv",
                            type="primary",
                            use_container_width=True
                        )
                        
                        st.success(f"💾 Results ready for download: {filename}")
                    else:
                        st.error("No results generated")
    
    def evaluate_on_test_data(self, model_name, test_df):
        st.info(f"🧪 Evaluating on {len(test_df):,} test records...")
        
        try:
            # Load predictor
            predictor = Dream11Predictor(
                model_dir=str(self.model_artifacts_dir),
                model_name=model_name
            )
            
            # Get unique matches
            matches = test_df.groupby(['match_id', 'date']).first().reset_index()
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (_, match) in enumerate(matches.iterrows()):
                progress = (idx + 1) / len(matches)
                progress_bar.progress(progress)
                status_text.text(f"Evaluating match {idx + 1}/{len(matches)}...")
                
                match_data = test_df[test_df['match_id'] == match['match_id']].copy()
                
                if len(match_data) < 11:
                    continue
                
                # Predict
                predictions = predictor.predict(match_data)
                predictions = predictions.sort_values('predicted_fantasy_points', ascending=False)
                
                # Get actual dream team (top 11 by actual fantasy points)
                actual_dream_team = match_data.nlargest(11, 'fantasy_points')
                
                # Get predicted dream team (top 11 by predicted points)
                predicted_dream_team = predictions.nlargest(11, 'predicted_fantasy_points')
                
                # Calculate MAE
                actual_total = actual_dream_team['fantasy_points'].sum()
                predicted_total = predicted_dream_team['predicted_fantasy_points'].sum()
                mae = abs(actual_total - predicted_total)
                
                result = {
                    'Match_ID': match['match_id'],
                    'Match_Date': match['date'].strftime('%Y-%m-%d'),
                    'Team_1': match.get('team', 'Unknown'),
                    'Team_2': match.get('opposition', 'Unknown'),
                    'Venue': match.get('venue', 'Unknown'),
                    'Match_Type': match.get('match_type', 'Unknown'),
                    'Dream_Team_Total_Points': round(actual_total, 2),
                    'Predicted_Team_Total': round(predicted_total, 2),
                    'MAE': round(mae, 2)
                }
                
                # Add predicted team players
                for i in range(min(11, len(predicted_dream_team))):
                    pred_player = predicted_dream_team.iloc[i]
                    result[f'Predicted_Player_{i+1}'] = pred_player['player']
                    result[f'Predicted_Player_{i+1}_Points'] = round(pred_player['predicted_fantasy_points'], 2)
                    result[f'Predicted_Player_{i+1}_Role'] = pred_player.get('role', 'Unknown')
                
                # Add actual dream team players
                for i in range(min(11, len(actual_dream_team))):
                    actual_player = actual_dream_team.iloc[i]
                    result[f'Dream_Team_Player_{i+1}'] = actual_player['player']
                    result[f'Dream_Team_Player_{i+1}_Points'] = round(actual_player['fantasy_points'], 2)
                    result[f'Dream_Team_Player_{i+1}_Role'] = actual_player.get('role', 'Unknown')
                
                results.append(result)
            
            progress_bar.progress(1.0)
            status_text.text("✓ Evaluation complete!")
            
            if len(results) == 0:
                st.error("No valid matches found for evaluation")
                return None
            
            results_df = pd.DataFrame(results)
            return results_df
            
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

def main():
    st.set_page_config(
        page_title="Model Evaluation UI",
        page_icon="🔬",
        layout="wide"
    )
    
    ui = ModelUI()
    ui.run()

if __name__ == '__main__':
    main()