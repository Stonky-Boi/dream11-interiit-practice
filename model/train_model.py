import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(model_type, data_path='data/processed/final_model_data.csv', artifacts_path='model_artifacts'):
    """
    Trains one model (XGBoost or LightGBM) to predict the final fantasy_points score.
    """
    logging.info(f"Starting model training for: {model_type}")
    
    try:
        df = pd.read_csv(data_path, parse_dates=['date'])
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_path}. Please run the data pipeline first.")
        return
    
    TRAINING_CUTOFF_DATE = '2024-06-30'
    df_train = df[df['date'] <= TRAINING_CUTOFF_DATE].copy()
    logging.info(f"Training data filtered to dates on or before {TRAINING_CUTOFF_DATE}. Shape: {df_train.shape}")

    # --- CORRECTED: Target is fantasy_points, Features are all historical roll_* columns ---
    target = 'fantasy_points'
    features = [col for col in df_train.columns if col.startswith('roll_')]
    
    if not features:
        logging.error("No features found. Make sure feature engineering created 'roll_*' columns.")
        return
    
    logging.info(f"Training with {len(features)} historical features to predict '{target}'.")
    
    X_train = df_train[features]
    y_train = df_train[target]
    
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
                                 early_stopping_rounds=50, eval_metric="rmse", random_state=42)
        val_size = int(len(X_train) * 0.1)
        X_train_split, y_train_split = X_train.iloc[:-val_size], y_train.iloc[:-val_size]
        X_val, y_val = X_train.iloc[-val_size:], y_train.iloc[-val_size:]
        model.fit(X_train_split, y_train_split, eval_set=[(X_val, y_val)], verbose=False)
        
    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor(objective='regression', n_estimators=1000, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
    else:
        raise ValueError("Invalid model_type. Choose 'xgboost' or 'lightgbm'.")

    artifacts_dir = Path(artifacts_path)
    artifacts_dir.mkdir(exist_ok=True)
    model_filename = f"{model_type}_model.joblib"
    save_path = artifacts_dir / model_filename
    joblib.dump(model, save_path)
    
    logging.info(f"Model training complete. '{model_type}' model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a fantasy cricket prediction model.")
    parser.add_argument('--model_type', type=str, required=True, choices=['lightgbm', 'xgboost'])
    args = parser.parse_args()
    
    train_model(model_type=args.model_type)