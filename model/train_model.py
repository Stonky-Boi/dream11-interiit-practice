import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import argparse
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(model_type, data_path='data/processed/final_model_data.parquet', 
                artifacts_path='model_artifacts', gender='male'):
    """
    Train model for specific gender.
    """
    logging.info(f"Starting T20 {gender} cricket model training for: {model_type}")
    
    df = pd.read_parquet(data_path)
    
    # Filter by gender
    if 'gender' in df.columns:
        df = df[df['gender'] == gender].copy()
        logging.info(f"Training on {gender} cricket: {len(df)} records")
    
    # ... rest of training code

    logging.info(f"Loaded data shape: {df.shape}")
    
    # --- Filter for T20 ONLY ---
    if 'match_type' in df.columns:
        non_t20 = df[df['match_type'] != 'T20']
        if len(non_t20) > 0:
            logging.warning(f"Found {len(non_t20)} non-T20 records! Filtering them out...")
            df = df[df['match_type'] == 'T20'].copy()
        
        logging.info(f"Training on {len(df)} T20 match records")
        logging.info(f"Unique T20 matches: {df['match_id'].nunique()}")
        logging.info(f"Unique players: {df['player'].nunique()}")
        logging.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    else:
        logging.warning("No match_type column found. Assuming all data is T20.")
    
    # --- Enforce Training Date Cutoff ---
    TRAINING_CUTOFF_DATE = '2024-06-30'
    df_train = df[df['date'] <= TRAINING_CUTOFF_DATE].copy()
    logging.info(f"Training data filtered to dates on or before {TRAINING_CUTOFF_DATE}")
    logging.info(f"Training data shape: {df_train.shape}")
    
    if df_train.empty:
        logging.error("No training data after filtering!")
        return
    
    target = 'fantasy_points'
    
    roll_features = [col for col in df_train.columns if col.startswith('roll_')]
    
    # Additional features from enhanced feature engineering
    additional_features = [
        'weighted_fp_5', 'weighted_runs_5', 'form_trend', 'consistency_score',
        'recent_performance_ratio', 'match_count', 'days_since_last_match',
        'batting_importance', 'bowling_workload', 'allrounder_score',
        'team_avg_fp', 'team_avg_runs', 'team_avg_wickets', 'team_avg_catches',
        'player_team_fp_ratio', 'venue_avg_fp',
        'strike_rate', 'economy_rate', 'overs_bowled',
        'venue_encoded', 'team_encoded', 'city_encoded'
    ]
    
    additional_features = [f for f in additional_features if f in df_train.columns]
    
    features = roll_features + additional_features
    
    if not features:
        logging.error("No features found! Make sure feature engineering was run.")
        logging.error("Expected features like: roll_*, weighted_fp_5, etc.")
        return
    
    logging.info(f"Using {len(features)} features for training:")
    logging.info(f"  - Rolling features: {len(roll_features)}")
    logging.info(f"  - Additional features: {len(additional_features)}")
    
    if 'match_count' in df_train.columns:
        df_train = df_train[df_train['match_count'] >= 3].copy()
        logging.info(f"After removing cold start (match_count < 3): {df_train.shape}")
    
    X_train = df_train[features].fillna(0)
    y_train = df_train[target]
    
    logging.info(f"Final training set: {X_train.shape}")
    logging.info(f"Target (fantasy_points) - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
    
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=50,
            eval_metric='rmse',
            random_state=42
        )
        
        val_size = int(len(X_train) * 0.15)
        X_train_split, y_train_split = X_train[:-val_size], y_train[:-val_size]
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        
        logging.info(f"Training on {len(X_train_split)} samples, validating on {len(X_val)} samples")
        
        model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        logging.info(f"\n{'='*80}")
        logging.info("VALIDATION PERFORMANCE")
        logging.info(f"{'='*80}")
        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"MAE: {mae:.4f}")
        logging.info(f"R² Score: {r2:.4f}")
        logging.info(f"{'='*80}\n")
        
    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1
        )
        
        val_size = int(len(X_train) * 0.15)
        X_train_split, y_train_split = X_train[:-val_size], y_train[:-val_size]
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        
        logging.info(f"Training on {len(X_train_split)} samples, validating on {len(X_val)} samples")
        
        model.fit(X_train_split, y_train_split)
        
        # Evaluate
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        logging.info(f"\n{'='*80}")
        logging.info("VALIDATION PERFORMANCE")
        logging.info(f"{'='*80}")
        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"MAE: {mae:.4f}")
        logging.info(f"R² Score: {r2:.4f}")
        logging.info(f"{'='*80}\n")
        
    else:
        raise ValueError("Invalid model_type. Choose 'xgboost' or 'lightgbm'.")
    
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logging.info("\nTop 15 Most Important Features:")
        logging.info(feature_importance.head(15).to_string(index=False))
    
    artifacts_dir = Path(artifacts_path)
    artifacts_dir.mkdir(exist_ok=True)
    
    model_artifact = {
        'model': model,
        'features': features,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        },
        'training_info': {
            'cutoff_date': TRAINING_CUTOFF_DATE,
            'training_samples': len(X_train_split),
            'validation_samples': len(X_val),
            'n_features': len(features),
            'match_type': 'T20'
        },
        'model_type': model_type
    }
    
    model_filename = f"{model_type}_model.joblib"
    save_path = artifacts_dir / model_filename
    joblib.dump(model_artifact, save_path)
    
    logging.info(f"\n✅ Model training complete!")
    logging.info(f"Model saved to: {save_path}")
    logging.info(f"Model type: {model_type}")
    logging.info(f"Features: {len(features)}")
    logging.info(f"Training samples: {len(X_train_split)}")
    logging.info(f"Validation RMSE: {rmse:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a T20 fantasy cricket prediction model.")
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=['lightgbm', 'xgboost'],
        help="The type of model to train (lightgbm or xgboost)."
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/processed/final_model_data.parquet',
        help="Path to processed data file"
    )
    parser.add_argument(
        '--artifacts_path',
        type=str,
        default='model_artifacts',
        help="Directory to save trained model"
    )
    
    args = parser.parse_args()
    
    train_model(
        model_type=args.model_type,
        data_path=args.data_path,
        artifacts_path=args.artifacts_path
    )
