"""
Prediction Module for Dream11 Fantasy Points
Makes predictions using trained ensemble models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Dream11Predictor:
    """Make predictions using trained ensemble models"""
    
    def __init__(self, model_dir='model_artifacts', model_name='ProductUIModel'):
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.models = {}
        self.ensemble_weights = {}
        self.metadata = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models and metadata"""
        print("=" * 70)
        print("LOADING MODELS")
        print("=" * 70)
        
        # Load metadata
        metadata_path = self.model_dir / f"{self.model_name}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_cols = self.metadata['feature_columns']
        self.categorical_features = self.metadata['categorical_features']
        
        print(f"✓ Model: {self.model_name}")
        print(f"✓ Training date: {self.metadata.get('training_date', 'Unknown')}")
        print(f"✓ Ensemble MAE: {self.metadata.get('ensemble_mae', 0):.2f}")
        print(f"✓ Features: {len(self.feature_cols)}")
        
        # Load ensemble weights
        weights_path = self.model_dir / f"{self.model_name}_ensemble_weights.json"
        with open(weights_path, 'r') as f:
            self.ensemble_weights = json.load(f)
        
        # Load individual models
        for model_type in ['xgboost', 'lightgbm', 'catboost']:
            model_path = self.model_dir / f"{self.model_name}_{model_type}.pkl"
            if model_path.exists():
                self.models[model_type] = joblib.load(model_path)
                print(f"✓ Loaded {model_type} (weight: {self.ensemble_weights.get(model_type, 0):.4f})")
        
        print(f"✓ Loaded {len(self.models)} models")
        print("=" * 70)
    
    def prepare_prediction_features(self, player_features_df):
        """Prepare features for prediction"""
        # Check for required columns
        missing_cols = [col for col in self.feature_cols if col not in player_features_df.columns]
        if missing_cols:
            print(f"⚠️ Warning: Missing {len(missing_cols)} features, filling with defaults")
            for col in missing_cols:
                if col in self.categorical_features:
                    player_features_df[col] = 'Unknown'
                else:
                    player_features_df[col] = 0
        
        # Ensure all required features are present
        X = player_features_df[self.feature_cols].copy()
        
        # Handle missing values and categorical features
        for col in self.feature_cols:
            if col in self.categorical_features:
                # For categorical columns, handle carefully
                if X[col].dtype.name != 'category':
                    # Convert to categorical, including 'Unknown' in categories
                    X[col] = X[col].fillna('Unknown').astype(str)
                    X[col] = pd.Categorical(X[col])
                else:
                    # Already categorical - add 'Unknown' to categories if needed
                    if 'Unknown' not in X[col].cat.categories:
                        X[col] = X[col].cat.add_categories(['Unknown'])
                    X[col] = X[col].fillna('Unknown')
            else:
                # For numerical columns
                median_val = X[col].median() if not X[col].isna().all() else 0
                X[col] = X[col].fillna(median_val)
        
        return X
    
    def predict(self, player_features_df):
        """Make ensemble predictions"""
        X = self.prepare_prediction_features(player_features_df)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            weight = self.ensemble_weights.get(name, 1.0 / len(self.models))
            ensemble_pred += weight * pred
        
        # Store individual predictions for explainability
        result_df = player_features_df.copy()
        result_df['predicted_fantasy_points'] = ensemble_pred
        
        for name, pred in predictions.items():
            result_df[f'pred_{name}'] = pred
        
        return result_df
    
    def predict_single_player(self, player_features):
        """Predict for a single player (dict input)"""
        df = pd.DataFrame([player_features])
        result = self.predict(df)
        return result.iloc[0]['predicted_fantasy_points']
    
    def get_feature_importance(self, model_type='xgboost', top_n=20):
        """Get feature importance from a specific model"""
        if model_type not in self.models:
            return None
        
        model = self.models[model_type]
        
        if model_type == 'xgboost':
            importance = model.feature_importances_
        elif model_type == 'lightgbm':
            importance = model.feature_importances_
        elif model_type == 'catboost':
            importance = model.feature_importances_
        else:
            return None
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return feature_importance_df
    
    def explain_prediction(self, player_row):
        """Generate explanation for a single prediction"""
        explanations = {
            'prediction': player_row.get('predicted_fantasy_points', 0),
            'recent_form': player_row.get('avg_fantasy_points_last_5', 0),
            'career_avg': player_row.get('career_avg_fantasy_points', 0),
            'venue_avg': player_row.get('venue_avg_fantasy_points', 0),
            'opposition_avg': player_row.get('opp_avg_fantasy_points', 0),
            'role': player_row.get('role', 'Unknown'),
            'matches_played': player_row.get('career_matches', 0)
        }
        return explanations

def main():
    """Test prediction module"""
    predictor = Dream11Predictor()
    
    # Load some test data
    data_path = Path('data/processed/training_data_2024-06-30.csv')
    if data_path.exists():
        df = pd.read_csv(data_path)
        print("\nTesting predictions on sample data...")
        
        sample = df.sample(10)
        predictions = predictor.predict(sample)
        
        print("\nSample Predictions:")
        print(predictions[['player', 'predicted_fantasy_points', 'fantasy_points']].head())
        
        # Feature importance
        print("\nTop 10 Features:")
        importance = predictor.get_feature_importance(top_n=10)
        print(importance)

if __name__ == '__main__':
    main()