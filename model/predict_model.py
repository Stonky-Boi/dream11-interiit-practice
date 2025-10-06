import pandas as pd
import joblib
import shap
from pathlib import Path
import logging
import argparse

class ModelPredictor:
    def __init__(self, model_path):
        model_p = Path(model_path)
        if not model_p.exists(): raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_p)
        logging.info(f"Model loaded successfully from {model_path}")
        self.explainer = shap.TreeExplainer(self.model)
        if hasattr(self.model, 'feature_name_'): self.features = self.model.feature_name_
        elif hasattr(self.model, 'feature_names_in_'): self.features = self.model.feature_names_in_
        else: self.features = None
    def predict(self, input_data):
        if self.features is not None: input_data = input_data[self.features]
        return self.model.predict(input_data)
    def explain_prediction(self, input_data):
        if self.features is not None: input_data = input_data[self.features]
        shap_values = self.explainer.shap_values(input_data)
        return {feature: val for feature, val in zip(self.features, shap_values[0])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the ModelPredictor class.")
    parser.add_argument('--model_type', type=str, default='lightgbm', choices=['lightgbm', 'xgboost'])
    args = parser.parse_args()
    
    MODEL_PATH = f'model_artifacts/{args.model_type}_model.joblib'
    PROCESSED_DATA_PATH = 'data/processed/final_model_data.csv'
    TRAINING_CUTOFF_DATE = '2024-06-30'
    
    try:
        predictor = ModelPredictor(model_path=MODEL_PATH)
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['date'])
        unseen_data = df[df['date'] > TRAINING_CUTOFF_DATE].copy()
        
        if unseen_data.empty:
            logging.warning("No unseen data found after cutoff. Cannot perform a real test.")
        else:
            sample_data = unseen_data.sample(n=min(5, len(unseen_data)), random_state=42)
            predictions = predictor.predict(sample_data)
            results = sample_data[['player', 'date', 'fantasy_points']].copy()
            results.rename(columns={'fantasy_points': 'Actual_Points'}, inplace=True)
            results['Predicted_Points'] = predictions
            print("\n--- ✅ Prediction Test Results (on unseen data) ---\n")
            print(results)
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