import pandas as pd
import joblib
import shap
from pathlib import Path
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelPredictor:
    def __init__(self, model_path):
        """
        Initializes the predictor by loading the model and creating a SHAP explainer.
        
        Args:
            model_path (str): Path to the trained model artifact.
        """
        model_p = Path(model_path)
        if not model_p.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.model = joblib.load(model_p)
        logging.info(f"Model loaded successfully from {model_path}")
        
        self.explainer = shap.TreeExplainer(self.model)
        
        if hasattr(self.model, 'feature_name_'): # LightGBM
            self.features = self.model.feature_name_
        elif hasattr(self.model, 'feature_names_in_'): # XGBoost
            self.features = self.model.feature_names_in_
        else:
            self.features = None
            logging.warning("Could not automatically determine feature names from the model.")

    def predict(self, input_data):
        """
        Makes a prediction on new input data.
        """
        if self.features is not None:
            input_data = input_data[self.features]
        return self.model.predict(input_data)

    def explain_prediction(self, input_data):
        """
        Generates SHAP values to explain a prediction.
        """
        if self.features is not None:
            input_data = input_data[self.features]
            
        shap_values = self.explainer.shap_values(input_data)
        
        explanation = {feature: shap_val for feature, shap_val in zip(self.features, shap_values[0])}
        return explanation


if __name__ == '__main__':
    # --- Standalone Test for the Predictor ---
    # This block allows you to test the model directly from the command line.
    
    parser = argparse.ArgumentParser(description="Test the ModelPredictor class.")
    parser.add_argument('--model_type', type=str, default='lightgbm', choices=['lightgbm', 'xgboost'],
                        help="The model artifact to test ('lightgbm' or 'xgboost').")
    args = parser.parse_args()

    MODEL_PATH = f'model_artifacts/{args.model_type}_model.joblib'
    PROCESSED_DATA_PATH = 'data/processed/final_model_data.parquet'
    TRAINING_CUTOFF_DATE = '2024-06-30'
    
    try:
        # 1. Initialize the predictor with a trained model
        predictor = ModelPredictor(model_path=MODEL_PATH)

        # 2. Load the full processed dataset to find unseen data for testing
        df = pd.read_parquet(PROCESSED_DATA_PATH)
        unseen_data = df[df['date'] > TRAINING_CUTOFF_DATE].copy()
        
        if unseen_data.empty:
            logging.warning("No unseen data found after the training cutoff date. Cannot perform a real test.")
        else:
            # 3. Select 5 random players from the unseen data
            sample_data = unseen_data.sample(n=min(5, len(unseen_data)), random_state=42)
            
            # 4. Get predictions
            predictions = predictor.predict(sample_data)
            
            # 5. Display results
            results = sample_data[['player', 'date', 'fantasy_points']].copy()
            results.rename(columns={'fantasy_points': 'Actual_Points'}, inplace=True)
            results['Predicted_Points'] = predictions
            
            print("\n--- ✅ Prediction Test Results (on unseen data) ---\n")
            print(results)
            
            # 6. Get a detailed explanation for the first player in the sample
            single_player_to_explain = sample_data.head(1)
            explanation = predictor.explain_prediction(single_player_to_explain)
            
            player_name = single_player_to_explain['player'].iloc[0]
            actual = single_player_to_explain['fantasy_points'].iloc[0]
            predicted = predictor.predict(single_player_to_explain)[0]

            print(f"\n--- ✨ Explanation for {player_name} ---")
            print(f"Actual Points: {actual:.2f} | Predicted Points: {predicted:.2f}\n")
            
            sorted_explanation = sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True)
            print("Top 5 Features Influencing Prediction:")
            for feature, value in sorted_explanation[:5]:
                print(f"  - {feature}: {'+' if value > 0 else ''}{value:.2f}")

    except FileNotFoundError as e:
        logging.error(f"ERROR: {e}")
        logging.error("Please train the model first by running `python model/train_model.py`")