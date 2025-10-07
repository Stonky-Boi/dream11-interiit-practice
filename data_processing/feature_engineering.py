import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_fantasy_points(df):
    """Calculate fantasy points using OFFICIAL Dream11 T20 rules."""
    logging.info("Calculating fantasy points with OFFICIAL Dream11 T20 rules...")
    
    df['fantasy_points'] = 0.0
    
    # BATTING POINTS
    df['fantasy_points'] += df['runs_scored'] * 1
    df['fantasy_points'] += df['fours'] * 4
    df['fantasy_points'] += df['sixes'] * 6
    
    milestone_bonus = np.where(df['runs_scored'] >= 100, 16,
                      np.where(df['runs_scored'] >= 50, 8,
                      np.where(df['runs_scored'] >= 30, 4, 0)))
    df['fantasy_points'] += milestone_bonus
    
    duck_penalty = np.where((df['runs_scored'] == 0) & (df['balls_faced'] > 0), -2, 0)
    df['fantasy_points'] += duck_penalty
    
    df['strike_rate'] = np.where(df['balls_faced'] > 0, 
                                  (df['runs_scored'] / df['balls_faced']) * 100, 0)
    
    sr_bonus = np.where(
        df['balls_faced'] >= 10,
        np.where(df['strike_rate'] > 170, 6,
        np.where(df['strike_rate'] > 150, 4,
        np.where(df['strike_rate'] >= 130, 2,
        np.where((df['strike_rate'] >= 60) & (df['strike_rate'] <= 70), -2,
        np.where((df['strike_rate'] >= 50) & (df['strike_rate'] < 60), -4,
        np.where(df['strike_rate'] < 50, -6, 0)))))),
        0
    )
    df['fantasy_points'] += sr_bonus
    
    # BOWLING POINTS
    df['fantasy_points'] += df['wickets'] * 30
    
    wicket_bonus = np.where(df['wickets'] >= 5, 16,
                   np.where(df['wickets'] == 4, 8,
                   np.where(df['wickets'] == 3, 4, 0)))
    df['fantasy_points'] += wicket_bonus
    
    df['overs_bowled'] = df['balls_bowled'] / 6.0
    maiden_overs = ((df['balls_bowled'] >= 6) & (df['runs_conceded'] == 0)).astype(int)
    df['fantasy_points'] += maiden_overs * 12
    
    df['economy_rate'] = np.where(df['overs_bowled'] > 0,
                                   df['runs_conceded'] / df['overs_bowled'], 0)
    
    er_bonus = np.where(
        df['overs_bowled'] >= 2,
        np.where(df['economy_rate'] < 5, 6,
        np.where(df['economy_rate'] < 6, 4,
        np.where(df['economy_rate'] <= 7, 2,
        np.where((df['economy_rate'] >= 10) & (df['economy_rate'] <= 11), -2,
        np.where((df['economy_rate'] > 11) & (df['economy_rate'] <= 12), -4,
        np.where(df['economy_rate'] > 12, -6, 0)))))),
        0
    )
    df['fantasy_points'] += er_bonus
    
    # FIELDING POINTS
    df['fantasy_points'] += df['catches'] * 8
    catch_bonus = np.where(df['catches'] >= 3, 4, 0)
    df['fantasy_points'] += catch_bonus
    
    if 'stumpings' in df.columns:
        df['fantasy_points'] += df['stumpings'] * 12
    
    if 'run_outs' in df.columns:
        df['fantasy_points'] += df['run_outs'] * 6
    
    logging.info("Fantasy points calculation complete.")
    return df


def create_rolling_features(df):
    """Enhanced rolling features."""
    logging.info("Creating rolling features...")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['player', 'date']).reset_index(drop=True)
    
    grouped = df.groupby('player')
    
    windows = [3, 5, 10]
    stats_to_roll = ['fantasy_points', 'runs_scored', 'wickets']
    
    for stat in stats_to_roll:
        for w in windows:
            df[f'roll_{stat}_{w}'] = grouped[stat].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            ).fillna(0)
            
            df[f'roll_{stat}_{w}_std'] = grouped[stat].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).std()
            ).fillna(0)
    
    def weighted_rolling_mean(series, window=5):
        weights = np.arange(1, window + 1)
        return series.shift(1).rolling(window, min_periods=1).apply(
            lambda x: np.average(x, weights=weights[:len(x)]) if len(x) > 0 else 0,
            raw=True
        )
    
    df['weighted_fp_5'] = grouped['fantasy_points'].transform(
        lambda x: weighted_rolling_mean(x, 5)
    ).fillna(0)
    
    df['form_trend'] = (
        df['roll_fantasy_points_3'] - 
        grouped['fantasy_points'].transform(
            lambda x: x.shift(4).rolling(3, min_periods=1).mean()
        )
    ).fillna(0)
    
    df['consistency_score'] = np.where(
        df['roll_fantasy_points_5'] > 0,
        1 / (1 + df['roll_fantasy_points_5_std'] / (df['roll_fantasy_points_5'] + 0.1)),
        0
    )
    
    logging.info("Rolling features created.")
    return df


def create_venue_features(df):
    """Venue-based features."""
    logging.info("Creating venue features...")
    
    venue_avg = df.groupby(['player', 'venue'])['fantasy_points'].mean().reset_index()
    venue_avg = venue_avg.rename(columns={'fantasy_points': 'venue_avg_fp'})
    
    df = pd.merge(df, venue_avg, on=['player', 'venue'], how='left')
    
    df['venue_avg_fp'] = df.groupby(['player', 'venue'])['venue_avg_fp'].transform(
        lambda x: x.shift(1)
    ).fillna(0)
    
    logging.info("Venue features created.")
    return df


def create_contextual_features(df):
    """Match frequency and experience features."""
    logging.info("Creating contextual features...")
    
    df = df.sort_values(by=['player', 'date']).reset_index(drop=True)
    
    df['match_count'] = df.groupby('player').cumcount() + 1
    df['days_since_last_match'] = df.groupby('player')['date'].diff().dt.days.fillna(7)
    
    logging.info("Contextual features created.")
    return df


def create_categorical_encodings(df):
    """Encode categorical variables."""
    logging.info("Creating categorical encodings...")
    
    if 'venue' in df.columns:
        df['venue_encoded'] = df['venue'].astype('category').cat.codes
    else:
        df['venue_encoded'] = 0
    
    if 'team' in df.columns:
        df['team_encoded'] = df['team'].astype('category').cat.codes
    else:
        df['team_encoded'] = 0
    
    if 'city' in df.columns:
        df['city_encoded'] = df['city'].astype('category').cat.codes
    else:
        df['city_encoded'] = 0
    
    logging.info("Categorical encodings created.")
    return df


if __name__ == '__main__':
    INTERIM_DATA_PATH = 'data/interim/player_match_stats.csv'  # CHANGED to .csv
    PROCESSED_DATA_PATH = 'data/processed/final_model_data.csv'  # CHANGED to .csv
    
    interim_path = Path(INTERIM_DATA_PATH)
    if not interim_path.exists():
        logging.error(f"{INTERIM_DATA_PATH} not found.")
    else:
        df = pd.read_csv(interim_path)  # CHANGED to read_csv
        
        # FILTER BY GENDER (optional - set to 'male' or 'female' or comment out)
        if 'gender' in df.columns:
            df = df[df['gender'] == 'male'].copy()  # Change to 'female' if needed
            logging.info(f"Filtered for male cricket: {len(df)} records")
        
        df = calculate_fantasy_points(df)
        df = create_rolling_features(df)
        df = create_venue_features(df)
        df = create_contextual_features(df)
        df = create_categorical_encodings(df)
        
        processed_path = Path(PROCESSED_DATA_PATH)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)  # CHANGED to to_csv
        
        logging.info(f"Feature engineering complete. Saved to {processed_path}")
