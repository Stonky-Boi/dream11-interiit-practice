import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_fantasy_points(df):
    """Calculates fantasy points for each player based on a defined scoring system."""
    batting_points = (df['runs_scored'] * 1 + df['fours'] * 1 + df['sixes'] * 2)
    batting_points += df['runs_scored'].apply(lambda x: 8 if x >= 50 else 0)
    batting_points += df['runs_scored'].apply(lambda x: 16 if x >= 100 else 0)
    bowling_points = df['wickets'] * 25
    bowling_points += df['wickets'].apply(lambda x: 8 if x >= 4 else 0)
    bowling_points += df['wickets'].apply(lambda x: 16 if x >= 5 else 0)
    fielding_points = (df['catches'] * 8) + (df['stumpings'] * 12) + (df['run_outs'] * 6)
    df['fantasy_points'] = batting_points + bowling_points + fielding_points
    return df

def create_rolling_features(df):
    """Engineers time-based rolling features for multiple raw performance stats."""
    logging.info("Creating rolling features for historical stats...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['player', 'date']).reset_index(drop=True)
    grouped = df.groupby('player')
    stats_to_roll = ['runs_scored', 'balls_faced', 'wickets', 'runs_conceded', 'catches']
    windows = [3, 5, 10]
    for stat in stats_to_roll:
        for w in windows:
            df[f'roll_{stat}_{w}'] = grouped[stat].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            ).fillna(0)
    logging.info("Rolling features created successfully.")
    return df

if __name__ == '__main__':
    INTERIM_DATA_PATH = 'data/interim/player_match_stats.csv'
    PROCESSED_DATA_PATH = 'data/processed/final_model_data.csv'
    
    interim_path = Path(INTERIM_DATA_PATH)
    if not interim_path.exists():
        logging.error(f"{INTERIM_DATA_PATH} not found. Please run data_preprocessing.py first.")
    else:
        df = pd.read_csv(interim_path, parse_dates=['date'])
        df = calculate_fantasy_points(df)        
        df = create_rolling_features(df)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        logging.info(f"Feature engineering complete. Final dataset saved to {PROCESSED_DATA_PATH}")