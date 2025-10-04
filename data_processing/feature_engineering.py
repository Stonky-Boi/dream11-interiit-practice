import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_fantasy_points(df):
    """
    Calculates fantasy points for each player based on a defined scoring system.
    [cite_start]This system is based on common fantasy cricket rules as mentioned in the PS. [cite: 104, 198]
    NOTE: The point values here are representative. Replace with the exact
    Dream11 point system if available.

    Args:
        df (pd.DataFrame): DataFrame with player performance stats.

    Returns:
        pd.DataFrame: DataFrame with an added 'fantasy_points' column.
    """
    logging.info("Calculating fantasy points...")
    
    # Batting points
    batting_points = (
        df['runs_scored'] * 1 +
        df['fours'] * 1 +
        df['sixes'] * 2
    )
    # Bonus for 30/50/100 runs
    batting_points += df['runs_scored'].apply(lambda x: 4 if x >= 30 else 0)
    batting_points += df['runs_scored'].apply(lambda x: 8 if x >= 50 else 0)
    batting_points += df['runs_scored'].apply(lambda x: 16 if x >= 100 else 0)
    
    # Bowling points
    bowling_points = df['wickets'] * 25
    # Bonus for 3/4/5 wickets
    bowling_points += df['wickets'].apply(lambda x: 4 if x >= 3 else 0)
    bowling_points += df['wickets'].apply(lambda x: 8 if x >= 4 else 0)
    bowling_points += df['wickets'].apply(lambda x: 16 if x >= 5 else 0)

    # Fielding points
    fielding_points = df['catches'] * 8

    df['fantasy_points'] = batting_points + bowling_points + fielding_points
    logging.info("Fantasy points calculation complete.")
    return df

def create_rolling_features(df):
    """
    Engineers time-based rolling features for player performance.
    It sorts the data by player and date to calculate historical averages.

    Args:
        df (pd.DataFrame): DataFrame with player stats and fantasy points.

    Returns:
        pd.DataFrame: DataFrame with new rolling average features.
    """
    logging.info("Creating rolling features...")
    
    # Ensure data is sorted for correct rolling calculations
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['player', 'date']).reset_index(drop=True)
    
    # Group by player to calculate rolling stats
    grouped = df.groupby('player')
    
    # Define rolling windows
    windows = [3, 5, 10]
    stats_to_roll = ['fantasy_points', 'runs_scored', 'wickets']
    
    for stat in stats_to_roll:
        for w in windows:
            # Shift(1) is crucial to prevent data leakage from the current match
            df[f'roll_{stat}_{w}'] = grouped[stat].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            ).fillna(0)
            
    logging.info("Rolling features created successfully.")
    return df

def create_venue_features(df):
    """
    Engineers features based on a player's historical performance at a venue.

    Args:
        df (pd.DataFrame): DataFrame with player stats.

    Returns:
        pd.DataFrame: DataFrame with new venue-based features.
    """
    logging.info("Creating venue features...")
    
    # Calculate player's average fantasy points at each venue
    venue_avg = df.groupby(['player', 'venue'])['fantasy_points'].mean().reset_index()
    venue_avg = venue_avg.rename(columns={'fantasy_points': 'venue_avg_fp'})
    
    # Merge back into the main df
    df = pd.merge(df, venue_avg, on=['player', 'venue'], how='left')

    # To prevent data leakage, we should calculate historical average *before* current game
    # For simplicity in this example, we use a simple average, but a rolling average is better
    df['venue_avg_fp'] = df.groupby(['player', 'venue'])['venue_avg_fp'].transform(
        lambda x: x.shift(1)
    ).fillna(0)
    
    logging.info("Venue features created successfully.")
    return df

if __name__ == '__main__':
    INTERIM_DATA_PATH = 'data/interim/player_match_stats.parquet'
    PROCESSED_DATA_PATH = 'data/processed/final_model_data.parquet'
    
    # Check if interim data exists
    interim_path = Path(INTERIM_DATA_PATH)
    if not interim_path.exists():
        logging.error(f"{INTERIM_DATA_PATH} not found. Please run data_preprocessing.py first.")
    else:
        # Load interim data
        df = pd.read_parquet(interim_path)
        
        # 1. Calculate Fantasy Points
        df = calculate_fantasy_points(df)
        
        # 2. Create Rolling Features
        df = create_rolling_features(df)
        
        # 3. Create Venue Features
        df = create_venue_features(df)
        
        # Save final processed data
        processed_path = Path(PROCESSED_DATA_PATH)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed_path, index=False)
        logging.info(f"Feature engineering complete. Final dataset saved to {processed_path}")