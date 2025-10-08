"""
Model Training Script for Dream11 Fantasy Points Prediction
Trains ensemble models + baseline comparisons
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

class Dream11ModelTrainer:
    """Train ensemble models for fantasy points prediction"""
    
    def __init__(self, data_path, model_artifacts_dir='model_artifacts'):
        self.data_path = Path(data_path)
        self.model_dir = Path(model_artifacts_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature columns (will be determined from data)
        self.feature_cols = None
        self.target_col = 'fantasy_points'
        self.categorical_features = ['role', 'match_type', 'gender']
        
        # Models (main + baselines)
        self.models = {}
        self.baseline_models = {}
        self.ensemble_weights = {}
        self.all_results = {}
    
    def load_data(self, train_end_date='2024-06-30'):
        """Load training data with date filtering"""
        print("=" * 70)
        print("LOADING TRAINING DATA")
        print("=" * 70)
        
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Filter by training end date (STRICT REQUIREMENT)
        train_end = pd.to_datetime(train_end_date)
        initial_count = len(self.df)
        self.df = self.df[self.df['date'] <= train_end]
        
        filtered_count = initial_count - len(self.df)
        
        print(f"✓ Loaded data up to {train_end_date}")
        print(f"✓ Total samples: {len(self.df):,}")
        print(f"✓ Filtered out (after cutoff): {filtered_count:,}")
        print(f"✓ Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        # Show match type distribution
        if 'match_type' in self.df.columns:
            print(f"✓ Match types: {self.df['match_type'].value_counts().to_dict()}")
        
        # Verify no data leakage
        if self.df['date'].max() > train_end:
            raise ValueError(f"⚠️ DATA LEAKAGE DETECTED! Found data after {train_end_date}")
        
        return self.df
    
    def prepare_features(self):
        """Prepare feature matrix and target"""
        print("\n" + "=" * 70)
        print("PREPARING FEATURES")
        print("=" * 70)
        
        # Remove rows with insufficient historical data
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['avg_fantasy_points_last_5'])
        print(f"✓ Removed {initial_count - len(self.df):,} rows with insufficient history")
        
        # Define feature columns
        feature_candidates = [
            # Historical performance (primary features)
            'avg_fantasy_points_last_3', 'avg_fantasy_points_last_5', 'avg_fantasy_points_last_10',
            'ema_fantasy_points', 'career_avg_fantasy_points',
            
            # Batting features
            'avg_runs_last_3', 'avg_runs_last_5', 'avg_runs_last_10',
            'avg_strike_rate_last_3', 'avg_strike_rate_last_5', 'avg_strike_rate_last_10',
            'career_avg_runs', 'career_avg_strike_rate',
            
            # Bowling features
            'avg_wickets_last_3', 'avg_wickets_last_5', 'avg_wickets_last_10',
            'avg_economy_last_3', 'avg_economy_last_5', 'avg_economy_last_10',
            'career_avg_wickets', 'career_avg_economy',
            
            # Venue features
            'venue_avg_fantasy_points', 'venue_matches',
            'venue_avg_runs', 'venue_avg_wickets',
            
            # Opposition features
            'opp_avg_fantasy_points', 'opp_matches',
            'opp_avg_runs', 'opp_avg_wickets',
            
            # Form and context
            'form_trend', 'consistency_last_5', 'career_matches',
            'days_since_last_match', 'month', 'year',
            
            # Toss features
            'won_toss', 'toss_bat',
            
            # Categorical (includes match_type for ODI/T20 differentiation)
            'role', 'match_type', 'gender'
        ]
        
        # Filter to available columns
        self.feature_cols = [col for col in feature_candidates if col in self.df.columns]
        
        print(f"✓ Selected {len(self.feature_cols)} features")
        print(f"  - Categorical: {[f for f in self.feature_cols if f in self.categorical_features]}")
        print(f"  - Numerical: {len([f for f in self.feature_cols if f not in self.categorical_features])}")
        
        # Handle missing values
        for col in self.feature_cols:
            if col not in self.categorical_features:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
            else:
                self.df[col].fillna('Unknown', inplace=True)
        
        # Encode categorical features
        for cat_col in self.categorical_features:
            if cat_col in self.df.columns:
                self.df[cat_col] = self.df[cat_col].astype('category')
        
        # Prepare X and y
        self.X = self.df[self.feature_cols].copy()
        self.y = self.df[self.target_col].copy()
        
        print(f"✓ Feature matrix shape: {self.X.shape}")
        print(f"✓ Target shape: {self.y.shape}")
        print(f"✓ Target statistics:")
        print(f"  - Mean: {self.y.mean():.2f}")
        print(f"  - Median: {self.y.median():.2f}")
        print(f"  - Std: {self.y.std():.2f}")
        print(f"  - Range: [{self.y.min():.2f}, {self.y.max():.2f}]")
        
        return self.X, self.y
    
    def split_data(self, test_size=0.15, random_state=42):
        """Split data into train and validation sets"""
        print("\n" + "=" * 70)
        print("SPLITTING DATA")
        print("=" * 70)
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        print(f"✓ Training set: {self.X_train.shape[0]:,} samples")
        print(f"✓ Validation set: {self.X_val.shape[0]:,} samples")
        print(f"✓ Split ratio: {(1-test_size)*100:.0f}% train / {test_size*100:.0f}% validation")
        
        return self.X_train, self.X_val, self.y_train, self.y_val
    
    def train_baseline_models(self):
        """Train baseline models for comparison"""
        print("\n" + "=" * 70)
        print("TRAINING BASELINE MODELS")
        print("=" * 70)
        
        # Convert categorical to numeric for baseline models
        X_train_numeric = self.X_train.copy()
        X_val_numeric = self.X_val.copy()
        
        for cat_col in self.categorical_features:
            if cat_col in X_train_numeric.columns:
                X_train_numeric[cat_col] = X_train_numeric[cat_col].cat.codes
                X_val_numeric[cat_col] = X_val_numeric[cat_col].cat.codes
        
        baseline_configs = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'lasso': Lasso(alpha=1.0, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        }
        
        for name, model in baseline_configs.items():
            print(f"\n[{name.upper().replace('_', ' ')}] Training...")
            
            model.fit(X_train_numeric, self.y_train)
            
            train_pred = model.predict(X_train_numeric)
            val_pred = model.predict(X_val_numeric)
            
            train_mae = mean_absolute_error(self.y_train, train_pred)
            val_mae = mean_absolute_error(self.y_val, val_pred)
            val_r2 = r2_score(self.y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
            
            print(f"  Train MAE: {train_mae:.2f}")
            print(f"  Val MAE:   {val_mae:.2f}")
            print(f"  Val RMSE:  {val_rmse:.2f}")
            print(f"  Val R²:    {val_r2:.4f}")
            
            self.baseline_models[name] = {
                'model': model,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'train_mae': train_mae
            }
            
            self.all_results[name] = {
                'type': 'baseline',
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'train_mae': train_mae
            }
    
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n" + "=" * 70)
        print("[XGBoost] TRAINING")
        print("=" * 70)
        
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            enable_categorical=True
        )
        
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False
        )
        
        # Predictions
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        
        train_mae = mean_absolute_error(self.y_train, train_pred)
        val_mae = mean_absolute_error(self.y_val, val_pred)
        val_r2 = r2_score(self.y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
        
        print(f"  Train MAE: {train_mae:.2f}")
        print(f"  Val MAE:   {val_mae:.2f}")
        print(f"  Val RMSE:  {val_rmse:.2f}")
        print(f"  Val R²:    {val_r2:.4f}")
        
        self.models['xgboost'] = {
            'model': model,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'train_mae': train_mae,
            'val_predictions': val_pred
        }
        
        self.all_results['xgboost'] = {
            'type': 'ensemble',
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'train_mae': train_mae
        }
        
        return model
    
    def train_lightgbm(self):
        """Train LightGBM model"""
        print("\n" + "=" * 70)
        print("[LightGBM] TRAINING")
        print("=" * 70)
        
        # Prepare categorical features for LightGBM
        cat_indices = [i for i, col in enumerate(self.X_train.columns) if col in self.categorical_features]
        
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            categorical_feature=cat_indices
        )
        
        # Predictions
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        
        train_mae = mean_absolute_error(self.y_train, train_pred)
        val_mae = mean_absolute_error(self.y_val, val_pred)
        val_r2 = r2_score(self.y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
        
        print(f"  Train MAE: {train_mae:.2f}")
        print(f"  Val MAE:   {val_mae:.2f}")
        print(f"  Val RMSE:  {val_rmse:.2f}")
        print(f"  Val R²:    {val_r2:.4f}")
        
        self.models['lightgbm'] = {
            'model': model,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'train_mae': train_mae,
            'val_predictions': val_pred
        }
        
        self.all_results['lightgbm'] = {
            'type': 'ensemble',
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'train_mae': train_mae
        }
        
        return model
    
    def train_catboost(self):
        """Train CatBoost model"""
        print("\n" + "=" * 70)
        print("[CatBoost] TRAINING")
        print("=" * 70)
        
        # Prepare categorical features
        cat_features = [col for col in self.categorical_features if col in self.X_train.columns]
        
        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False,
            cat_features=cat_features
        )
        
        model.fit(
            self.X_train, self.y_train,
            eval_set=(self.X_val, self.y_val),
            use_best_model=True
        )
        
        # Predictions
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        
        train_mae = mean_absolute_error(self.y_train, train_pred)
        val_mae = mean_absolute_error(self.y_val, val_pred)
        val_r2 = r2_score(self.y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
        
        print(f"  Train MAE: {train_mae:.2f}")
        print(f"  Val MAE:   {val_mae:.2f}")
        print(f"  Val RMSE:  {val_rmse:.2f}")
        print(f"  Val R²:    {val_r2:.4f}")
        
        self.models['catboost'] = {
            'model': model,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'train_mae': train_mae,
            'val_predictions': val_pred
        }
        
        self.all_results['catboost'] = {
            'type': 'ensemble',
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'train_mae': train_mae
        }
        
        return model
    
    def create_ensemble(self):
        """Create weighted ensemble of all models"""
        print("\n" + "=" * 70)
        print("CREATING ENSEMBLE")
        print("=" * 70)
        
        # Inverse MAE weighting (better models get higher weight)
        total_inverse_mae = sum(1/m['val_mae'] for m in self.models.values())
        
        print("Model weights (based on inverse MAE):")
        for name, model_dict in self.models.items():
            weight = (1/model_dict['val_mae']) / total_inverse_mae
            self.ensemble_weights[name] = weight
            print(f"  {name:12s}: {weight:.4f} (MAE: {model_dict['val_mae']:.2f})")
        
        # Ensemble predictions on validation set
        ensemble_pred = np.zeros(len(self.y_val))
        for name, model_dict in self.models.items():
            ensemble_pred += self.ensemble_weights[name] * model_dict['val_predictions']
        
        ensemble_mae = mean_absolute_error(self.y_val, ensemble_pred)
        ensemble_r2 = r2_score(self.y_val, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(self.y_val, ensemble_pred))
        
        # Calculate train predictions for ensemble
        ensemble_train_pred = np.zeros(len(self.y_train))
        for name, model_dict in self.models.items():
            train_pred = model_dict['model'].predict(self.X_train)
            ensemble_train_pred += self.ensemble_weights[name] * train_pred
        
        ensemble_train_mae = mean_absolute_error(self.y_train, ensemble_train_pred)
        
        print(f"\n{'='*70}")
        print("ENSEMBLE PERFORMANCE")
        print("=" * 70)
        print(f"  Train MAE: {ensemble_train_mae:.2f}")
        print(f"  Val MAE:   {ensemble_mae:.2f}")
        print(f"  Val RMSE:  {ensemble_rmse:.2f}")
        print(f"  Val R²:    {ensemble_r2:.4f}")
        
        self.ensemble_mae = ensemble_mae
        self.ensemble_rmse = ensemble_rmse
        self.ensemble_r2 = ensemble_r2
        self.ensemble_train_mae = ensemble_train_mae
        
        self.all_results['ensemble'] = {
            'type': 'ensemble',
            'val_mae': ensemble_mae,
            'val_rmse': ensemble_rmse,
            'val_r2': ensemble_r2,
            'train_mae': ensemble_train_mae
        }
        
        return ensemble_pred
    
    def compare_all_models(self):
        """Generate comparison report of all models"""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)
        
        # Sort by validation MAE
        sorted_models = sorted(self.all_results.items(), key=lambda x: x[1]['val_mae'])
        
        print(f"\n{'Model':<20} {'Type':<12} {'Train MAE':<12} {'Val MAE':<12} {'Val RMSE':<12} {'Val R²':<10}")
        print("-" * 90)
        
        for name, results in sorted_models:
            print(f"{name:<20} {results['type']:<12} "
                  f"{results['train_mae']:>10.2f}  "
                  f"{results['val_mae']:>10.2f}  "
                  f"{results['val_rmse']:>10.2f}  "
                  f"{results['val_r2']:>10.4f}")
        
        best_model = sorted_models[0][0]
        print(f"\n✓ Best Model: {best_model.upper()} (Val MAE: {sorted_models[0][1]['val_mae']:.2f})")
        
        # Calculate improvement over baseline
        if 'linear_regression' in self.all_results:
            baseline_mae = self.all_results['linear_regression']['val_mae']
            ensemble_mae = self.all_results['ensemble']['val_mae']
            improvement = ((baseline_mae - ensemble_mae) / baseline_mae) * 100
            print(f"✓ Improvement over Linear Regression: {improvement:.1f}%")
    
    def save_models(self, model_name='ProductUIModel'):
        """Save trained models and metadata"""
        print("\n" + "=" * 70)
        print("SAVING MODELS")
        print("=" * 70)
        
        # Save individual ensemble models
        for name, model_dict in self.models.items():
            model_path = self.model_dir / f"{model_name}_{name}.pkl"
            joblib.dump(model_dict['model'], model_path)
            print(f"✓ Saved {name:12s} -> {model_path.name}")
        
        # Save baseline models
        for name, model_dict in self.baseline_models.items():
            model_path = self.model_dir / f"{model_name}_baseline_{name}.pkl"
            joblib.dump(model_dict['model'], model_path)
        print(f"✓ Saved {len(self.baseline_models)} baseline models")
        
        # Save ensemble weights
        weights_path = self.model_dir / f"{model_name}_ensemble_weights.json"
        with open(weights_path, 'w') as f:
            json.dump(self.ensemble_weights, f, indent=2)
        print(f"✓ Saved ensemble weights -> {weights_path.name}")
        
        # Save complete comparison results
        comparison_path = self.model_dir / f"{model_name}_model_comparison.json"
        comparison_data = {
            'all_models': self.all_results,
            'best_model': min(self.all_results.items(), key=lambda x: x[1]['val_mae'])[0],
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"✓ Saved model comparison -> {comparison_path.name}")
        
        # Save feature metadata
        metadata = {
            'feature_columns': self.feature_cols,
            'categorical_features': self.categorical_features,
            'target_column': self.target_col,
            'ensemble_mae': float(self.ensemble_mae),
            'ensemble_rmse': float(self.ensemble_rmse),
            'ensemble_r2': float(self.ensemble_r2),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_samples': len(self.X_train),
            'validation_samples': len(self.X_val),
            'model_performances': {
                name: {
                    'val_mae': float(model_dict['val_mae']),
                    'val_rmse': float(model_dict['val_rmse']),
                    'val_r2': float(model_dict['val_r2'])
                }
                for name, model_dict in self.models.items()
            }
        }
        
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata -> {metadata_path.name}")
        
        print(f"\n✓ All models saved to: {self.model_dir}/")
    
    def train_full_pipeline(self, train_end_date='2024-06-30', model_name='ProductUIModel'):
        """Execute complete training pipeline"""
        print("\n" + "=" * 70)
        print("DREAM11 MODEL TRAINING PIPELINE")
        print("=" * 70)
        print(f"Model Name: {model_name}")
        print(f"Training Cutoff: {train_end_date}")
        print("=" * 70)
        
        # Load and prepare data
        self.load_data(train_end_date)
        self.prepare_features()
        self.split_data()
        
        # Train baseline models
        self.train_baseline_models()
        
        # Train ensemble models
        self.train_xgboost()
        self.train_lightgbm()
        self.train_catboost()
        
        # Create ensemble
        self.create_ensemble()
        
        # Compare all models
        self.compare_all_models()
        
        # Save models
        self.save_models(model_name)
        
        print("\n" + "=" * 70)
        print("✓✓✓ TRAINING COMPLETE ✓✓✓")
        print("=" * 70)
        print(f"\nBest Model: Ensemble (MAE: {self.ensemble_mae:.2f})")
        print(f"Models saved to: {self.model_dir}/")
        print("\nNext step: streamlit run main_app.py")

def main():
    """Main execution"""
    trainer = Dream11ModelTrainer(
        data_path='data/processed/training_data_2024-06-30.csv'
    )
    
    trainer.train_full_pipeline(
        train_end_date='2024-06-30',
        model_name='ProductUIModel'
    )

if __name__ == '__main__':
    main()