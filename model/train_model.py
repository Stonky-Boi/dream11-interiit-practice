import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(model_type, data_path='data/processed/final_model_data.parquet', artifacts_path='model_artifacts'):
    """
    Trains a model (XGBoost or LightGBM) on the processed data.
    Args:
        model_type (str): The type of model to train ('xgboost' or 'lightgbm').
        data_path (str): Path to the processed data.
        artifacts_path (str): Directory to save the trained model.
    """
    logging.info(f"Starting model training for: {model_type}")
    
    # --- 1. Load Data ---
    df = pd.read_parquet(data_path)
    
    # --- 2. Enforce Training Date Cutoff ---
    # This is a strict rule from the problem statement.
    TRAINING_CUTOFF_DATE = '2024-06-30'
    df_train = df[df['date'] <= TRAINING_CUTOFF_DATE].copy()
    logging.info(f"Training data filtered to dates on or before {TRAINING_CUTOFF_DATE}. Shape: {df_train.shape}")

    # --- 3. Define Features (X) and Target (y) ---
    target = 'fantasy_points'
    features = [col for col in df_train.columns if col.startswith('roll_')]
    if not features:
        logging.error("No features found. Make sure feature engineering was run and columns are named 'roll_...'.")
        return
    X_train = df_train[features]
    y_train = df_train[target]
    
    # --- 4. Train Model ---
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, early_stopping_rounds=50, eval_metric="rmse")
        # We split the data without shuffling to respect the time-series nature.
        # The last 10% of the training data will be used for validation.
        val_size = int(len(X_train) * 0.1)
        X_train_split, y_train_split = X_train[:-val_size], y_train[:-val_size]
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        logging.info(f"Training on {len(X_train_split)} samples, validating on {len(X_val)} samples.")
        model.fit(X_train_split, y_train_split, eval_set=[(X_val, y_val)], verbose=False)
        
    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor(objective='regression', n_estimators=1000, learning_rate=0.05)
        model.fit(X_train, y_train)
        
    else:
        raise ValueError("Invalid model_type. Choose 'xgboost' or 'lightgbm'.")

    # --- 5. Save Model ---
    artifacts_dir = Path(artifacts_path)
    artifacts_dir.mkdir(exist_ok=True)
    model_filename = f"{model_type}_model.joblib"
    save_path = artifacts_dir / model_filename
    joblib.dump(model, save_path)
    logging.info(f"Model training complete. '{model_type}' model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a fantasy cricket prediction model.")
    parser.add_argument('--model_type', type=str, choices=['lightgbm', 'xgboost'], help="The type of model to train (lightgbm or xgboost).")
    args = parser.parse_args()
    train_model(model_type=args.model_type)