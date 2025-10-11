"""
Model Training Script for Dream11 Fantasy Points Prediction
Temporal train/val/test split with comprehensive evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class Dream11ModelTrainer:
    """Train ensemble models with temporal validation"""
    
    def __init__(self, data_path, model_artifacts_dir='model_artifacts'):
        self.data_path = Path(data_path)
        self.model_dir = Path(model_artifacts_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_cols = None
        self.target_col = 'fantasy_points'
        self.categorical_features = ['role', 'match_type']
        self.id_columns = ['player', 'match_id', 'date', 'venue', 'team', 'opposition']  # Already has venue
        
        self.models = {}
        self.baseline_models = {}
        self.ensemble_weights = {}
        self.all_results = {}
    
    def load_data(self):
        """Load all available data"""
        print("=" * 100)
        print("LOADING DATA")
        print("=" * 100)
        
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        print(f"✓ Loaded {len(self.df):,} samples")
        print(f"✓ Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
        print(f"✓ Match types: {self.df['match_type'].value_counts().to_dict()}")
        
        return self.df
    
    def prepare_features(self):
        """Prepare feature matrix and target"""
        print("\n" + "=" * 100)
        print("PREPARING FEATURES")
        print("=" * 100)
        
        # Remove rows with insufficient history
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['avg_fantasy_points_last_5'])
        print(f"✓ Removed {initial_count - len(self.df):,} rows with insufficient history")
        
        # Identify feature columns - exclude IDs, target, and string columns
        exclude_cols = self.id_columns + [self.target_col, 'career_best_bowling', 'venue']  # Added 'venue'
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        print(f"✓ Selected {len(self.feature_cols)} features")
        
        # Handle missing values
        for col in self.feature_cols:
            if col not in self.categorical_features:
                # Check if column is numeric before calculating median
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                else:
                    # If non-numeric slipped through, exclude it
                    print(f"⚠️  Skipping non-numeric column: {col}")
                    self.feature_cols.remove(col)
            else:
                self.df[col].fillna('Unknown', inplace=True)
        
        # Encode categorical
        for cat_col in self.categorical_features:
            if cat_col in self.df.columns:
                self.df[cat_col] = self.df[cat_col].astype('category')
        
        self.X = self.df[self.feature_cols].copy()
        self.y = self.df[self.target_col].copy()
        
        print(f"✓ Final feature count: {len(self.feature_cols)}")
        
        return self.X, self.y
    
    def split_train_val_test(self, val_frac=0.1):
        """Temporal split: train/val/test"""
        print("\n" + "=" * 100)
        print("TEMPORAL SPLITTING: TRAIN/VAL/TEST")
        print("=" * 100)
        
        train_cutoff = pd.to_datetime('2024-06-30')
        test_start = pd.to_datetime('2024-07-01')
        
        train_df = self.df[self.df['date'] <= train_cutoff].sort_values('date')
        test_df = self.df[self.df['date'] >= test_start].sort_values('date')
        
        print(f"✓ Training samples (<= 2024-06-30): {len(train_df):,}")
        print(f"✓ Testing samples (>= 2024-07-01):  {len(test_df):,}")
        
        # Validate training cutoff
        if train_df['date'].max() > train_cutoff:
            raise ValueError(f"⚠️ DATA LEAKAGE: Training data exceeds cutoff!")
        
        # Temporal split of train into train/val
        n_train = len(train_df)
        n_val = int(val_frac * n_train)
        train_main_df = train_df.iloc[:n_train - n_val]
        val_df = train_df.iloc[n_train - n_val:]
        
        print(f"✓ Main training: {len(train_main_df):,} (up to {train_main_df['date'].max().date()})")
        print(f"✓ Validation:    {len(val_df):,} ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
        
        self.X_train = train_main_df[self.feature_cols]
        self.y_train = train_main_df[self.target_col]
        self.X_val = val_df[self.feature_cols]
        self.y_val = val_df[self.target_col]
        self.X_test = test_df[self.feature_cols]
        self.y_test = test_df[self.target_col]
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def train_baseline_models(self):
        """Train baseline models"""
        print("\n" + "=" * 100)
        print("TRAINING BASELINE MODELS")
        print("=" * 100)
        
        X_train_numeric = self.X_train.copy()
        X_val_numeric = self.X_val.copy()
        
        for cat_col in self.categorical_features:
            if cat_col in X_train_numeric.columns:
                X_train_numeric[cat_col] = X_train_numeric[cat_col].cat.codes
                X_val_numeric[cat_col] = X_val_numeric[cat_col].cat.codes
        
        baseline_configs = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'lasso': Lasso(alpha=1.0, random_state=42, max_iter=5000),
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        }
        
        for name, model in baseline_configs.items():
            print(f"\n[{name.upper()}] Training...")
            model.fit(X_train_numeric, self.y_train)
            
            train_pred = model.predict(X_train_numeric)
            val_pred = model.predict(X_val_numeric)
            
            train_mae = mean_absolute_error(self.y_train, train_pred)
            val_mae = mean_absolute_error(self.y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
            val_r2 = r2_score(self.y_val, val_pred)
            
            print(f"  Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | Val R²: {val_r2:.4f}")
            
            self.baseline_models[name] = {
                'model': model,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'train_mae': train_mae,
                'val_predictions': val_pred
            }
            
            self.all_results[name] = {
                'type': 'baseline',
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'train_mae': train_mae
            }
    
    def train_xgboost(self):
        """Train XGBoost"""
        print("\n" + "=" * 100)
        print("[XGBoost] TRAINING")
        print("=" * 100)
        
        model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, enable_categorical=True
        )
        
        model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
        
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        
        train_mae = mean_absolute_error(self.y_train, train_pred)
        val_mae = mean_absolute_error(self.y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
        val_r2 = r2_score(self.y_val, val_pred)
        
        print(f"  Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | Val R²: {val_r2:.4f}")
        
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
        """Train LightGBM"""
        print("\n" + "=" * 100)
        print("[LightGBM] TRAINING")
        print("=" * 100)
        
        cat_indices = [i for i, col in enumerate(self.X_train.columns) if col in self.categorical_features]
        
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1
        )
        
        model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], categorical_feature=cat_indices)
        
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        
        train_mae = mean_absolute_error(self.y_train, train_pred)
        val_mae = mean_absolute_error(self.y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
        val_r2 = r2_score(self.y_val, val_pred)
        
        print(f"  Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | Val R²: {val_r2:.4f}")
        
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
        """Train CatBoost"""
        print("\n" + "=" * 100)
        print("[CatBoost] TRAINING")
        print("=" * 100)
        
        cat_features = [col for col in self.categorical_features if col in self.X_train.columns]
        
        model = CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=7,
            l2_leaf_reg=3, random_seed=42, verbose=False,
            cat_features=cat_features
        )
        
        model.fit(self.X_train, self.y_train, eval_set=(self.X_val, self.y_val), use_best_model=True)
        
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        
        train_mae = mean_absolute_error(self.y_train, train_pred)
        val_mae = mean_absolute_error(self.y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
        val_r2 = r2_score(self.y_val, val_pred)
        
        print(f"  Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | Val R²: {val_r2:.4f}")
        
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
        """Create weighted ensemble"""
        print("\n" + "=" * 100)
        print("CREATING ENSEMBLE")
        print("=" * 100)
        
        total_inverse_mae = sum(1/m['val_mae'] for m in self.models.values())
        
        print("Model weights (inverse MAE):")
        for name, model_dict in self.models.items():
            weight = (1/model_dict['val_mae']) / total_inverse_mae
            self.ensemble_weights[name] = weight
            print(f"  {name:12s}: {weight:.4f} (MAE: {model_dict['val_mae']:.2f})")
        
        ensemble_val = np.zeros(len(self.y_val))
        for name, model_dict in self.models.items():
            ensemble_val += self.ensemble_weights[name] * model_dict['val_predictions']
        
        val_mae = mean_absolute_error(self.y_val, ensemble_val)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, ensemble_val))
        val_r2 = r2_score(self.y_val, ensemble_val)
        
        ensemble_train = np.zeros(len(self.y_train))
        for name, model_dict in self.models.items():
            train_pred = model_dict['model'].predict(self.X_train)
            ensemble_train += self.ensemble_weights[name] * train_pred
        
        train_mae = mean_absolute_error(self.y_train, ensemble_train)
        
        print(f"\nEnsemble: Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | Val R²: {val_r2:.4f}")
        
        self.ensemble_mae = val_mae
        self.ensemble_rmse = val_rmse
        self.ensemble_r2 = val_r2
        self.ensemble_train_mae = train_mae
        
        self.all_results['ensemble'] = {
            'type': 'ensemble',
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'train_mae': train_mae
        }
    
    def evaluate_on_test(self):
        """Evaluate all models on test set"""
        print("\n" + "=" * 100)
        print("EVALUATING ON TEST SET (>= 2024-07-01)")
        print("=" * 100)
        
        test_scores = {}
        
        if len(self.X_test) == 0:
            print("⚠️  No test data available")
            return test_scores, None
        
        # Prepare numeric test data for baseline models
        X_test_numeric = self.X_test.copy()
        for cat_col in self.categorical_features:
            if cat_col in X_test_numeric.columns:
                X_test_numeric[cat_col] = X_test_numeric[cat_col].cat.codes
        
        # Evaluate baseline models (using numeric data)
        for name, model_dict in self.baseline_models.items():
            model = model_dict['model']
            test_pred = model.predict(X_test_numeric)  # Use numeric version
            
            test_mae = mean_absolute_error(self.y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            test_r2 = r2_score(self.y_test, test_pred)
            
            print(f"[{name:20s}] Test MAE: {test_mae:.2f} | RMSE: {test_rmse:.2f} | R²: {test_r2:.4f}")
            
            model_dict['test_mae'] = test_mae
            model_dict['test_rmse'] = test_rmse
            model_dict['test_r2'] = test_r2
            test_scores[name] = (test_mae, test_rmse, test_r2)
        
        # Evaluate ensemble models (using original categorical data)
        for name, model_dict in self.models.items():
            model = model_dict['model']
            test_pred = model.predict(self.X_test)  # Use original with categories
            
            test_mae = mean_absolute_error(self.y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            test_r2 = r2_score(self.y_test, test_pred)
            
            print(f"[{name:20s}] Test MAE: {test_mae:.2f} | RMSE: {test_rmse:.2f} | R²: {test_r2:.4f}")
            
            model_dict['test_mae'] = test_mae
            model_dict['test_rmse'] = test_rmse
            model_dict['test_r2'] = test_r2
            test_scores[name] = (test_mae, test_rmse, test_r2)
        
        # Ensemble prediction
        ensemble_pred = np.zeros(len(self.y_test))
        for name in self.models:
            model = self.models[name]['model']
            weight = self.ensemble_weights[name]
            ensemble_pred += weight * model.predict(self.X_test)
        
        e_mae = mean_absolute_error(self.y_test, ensemble_pred)
        e_rmse = np.sqrt(mean_squared_error(self.y_test, ensemble_pred))
        e_r2 = r2_score(self.y_test, ensemble_pred)
        
        print(f"[{'ensemble':20s}] Test MAE: {e_mae:.2f} | RMSE: {e_rmse:.2f} | R²: {e_r2:.4f}")
        
        test_scores['ensemble'] = (e_mae, e_rmse, e_r2)
        
        return test_scores, ensemble_pred
    
    def compare_all_models(self):
        """Print model comparison"""
        print("\n" + "=" * 100)
        print("MODEL COMPARISON (VALIDATION SET)")
        print("=" * 100)
        
        sorted_models = sorted(self.all_results.items(), key=lambda x: x[1]['val_mae'])
        
        print(f"\n{'Model':<20} {'Type':<12} {'Train MAE':<12} {'Val MAE':<12} {'Val R²':<10}")
        print("-" * 70)
        
        for name, results in sorted_models:
            print(f"{name:<20} {results['type']:<12} "
                  f"{results['train_mae']:>10.2f}  {results['val_mae']:>10.2f}  {results['val_r2']:>10.4f}")
        
        best_model = sorted_models[0][0]
        print(f"\n✓ Best Model: {best_model.upper()} (Val MAE: {sorted_models[0][1]['val_mae']:.2f})")
    
    def plot_and_save_metrics(self, test_scores):
        """Save plots to docs/"""
        docs_dir = Path("docs")
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        names = list(test_scores.keys())
        maes = [test_scores[name][0] for name in names]
        
        plt.figure(figsize=(10, 6))
        plt.bar(names, maes, color='skyblue', edgecolor='navy')
        plt.ylabel("MAE", fontsize=12)
        plt.xlabel("Model", fontsize=12)
        plt.title("Test Set MAE by Model", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(docs_dir / "model_test_mae_comparison.png", dpi=300)
        plt.close()
        
        print(f"✓ Saved plot: {docs_dir}/model_test_mae_comparison.png")
    
    def plot_ensemble_preds_vs_actual(self, ensemble_pred):
        """Scatter plot: predicted vs actual"""
        docs_dir = Path("docs")
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(self.y_test, ensemble_pred, alpha=0.3, s=10)
        plt.xlabel("Actual Fantasy Points", fontsize=12)
        plt.ylabel("Predicted Fantasy Points", fontsize=12)
        plt.title("Test Set: Ensemble Prediction vs Actual", fontsize=14, fontweight='bold')
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), ensemble_pred.min())
        max_val = max(self.y_test.max(), ensemble_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(docs_dir / "ensemble_preds_vs_actual.png", dpi=300)
        plt.close()
        
        print(f"✓ Saved plot: {docs_dir}/ensemble_preds_vs_actual.png")
    
    def save_models(self, model_name='ProductUIModel'):
        """Save all trained models and metadata"""
        print("\n" + "=" * 100)
        print("SAVING MODELS")
        print("=" * 100)
        
        # Save ensemble models
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
        print(f"✓ Saved ensemble weights")
        
        # Save model comparison
        comparison_path = self.model_dir / f"{model_name}_model_comparison.json"
        comparison_data = {
            'all_models': self.all_results,
            'best_model': min(self.all_results.items(), key=lambda x: x[1]['val_mae'])[0],
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"✓ Saved model comparison")
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_cols,
            'categorical_features': self.categorical_features,
            'target_column': self.target_col,
            'num_features': len(self.feature_cols),
            'ensemble_mae': float(self.ensemble_mae),
            'ensemble_rmse': float(self.ensemble_rmse),
            'ensemble_r2': float(self.ensemble_r2),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_samples': len(self.X_train),
            'validation_samples': len(self.X_val),
            'testing_samples': len(self.X_test) if hasattr(self, 'X_test') else 0
        }
        
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata")
        
        print(f"\n✓ All artifacts saved to: {self.model_dir}/")
    
    def train_full_pipeline(self, model_name='ProductUIModel'):
        """Execute complete training pipeline"""
        print("\n" + "=" * 100)
        print("DREAM11 TRAINING PIPELINE (TEMPORAL VALIDATION)")
        print("=" * 100)
        
        self.load_data()
        self.prepare_features()
        self.split_train_val_test(val_frac=0.1)
        self.train_baseline_models()
        self.train_xgboost()
        self.train_lightgbm()
        self.train_catboost()
        self.create_ensemble()
        self.compare_all_models()
        
        test_scores, ensemble_pred = self.evaluate_on_test()
        
        if test_scores and ensemble_pred is not None:
            self.plot_and_save_metrics(test_scores)
            self.plot_ensemble_preds_vs_actual(ensemble_pred)
        
        self.save_models(model_name)
        
        print("\n" + "=" * 100)
        print("✓✓✓ TRAINING COMPLETE ✓✓✓")
        print("=" * 100)
        print(f"\nBest Model: Ensemble (Val MAE: {self.ensemble_mae:.2f})")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Artifacts: {self.model_dir}/")
        print(f"Plots: docs/")

def main():
    trainer = Dream11ModelTrainer(
        data_path='data/processed/training_data_all.csv'
    )
    trainer.train_full_pipeline(model_name='ProductUIModel')

if __name__ == '__main__':
    main()