import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_fantasy_points(df):
    """
    Calculate fantasy points using OFFICIAL Dream11 T20 scoring rules.
    Source: https://www.dream11.com/games/point-system
    
    FIXED from original:
    - Wickets: 25 → 30 points (official Dream11)
    - 5-wicket bonus: 12 → 16 points (official Dream11)
    - Added: Strike rate bonuses/penalties
    - Added: Economy rate bonuses/penalties
    - Added: Duck penalty (-2)
    - Added: Maiden overs (+12)
    - Added: 3-catch bonus (+4)
    """
    logging.info("Calculating fantasy points with OFFICIAL Dream11 T20 rules...")
    
    # Initialize points column
    df['fantasy_points'] = 0.0
    
    # ===== BATTING POINTS =====
    # Base runs: +1 per run
    df['fantasy_points'] += df['runs_scored'] * 1
    
    # Boundary bonus: +4 TOTAL (run already counted, so we add the 4 runs scored as boundary)
    # Dream11 gives +4 for boundary, +6 for six (these are TOTAL points including the runs)
    # Since runs_scored already includes the 4 or 6, we ADD the bonus portion
    df['fantasy_points'] += df['fours'] * 4  # +4 bonus per four
    df['fantasy_points'] += df['sixes'] * 6  # +6 bonus per six
    
    # Milestone bonuses (exclusive - only highest applies)
    milestone_bonus = np.where(df['runs_scored'] >= 100, 16,
                      np.where(df['runs_scored'] >= 50, 8,
                      np.where(df['runs_scored'] >= 30, 4, 0)))
    df['fantasy_points'] += milestone_bonus
    
    # Duck penalty: -2 for scoring 0 after facing at least 1 ball
    duck_penalty = np.where((df['runs_scored'] == 0) & (df['balls_faced'] > 0), -2, 0)
    df['fantasy_points'] += duck_penalty
    
    # Calculate strike rate for SR bonuses
    df['strike_rate'] = np.where(df['balls_faced'] > 0, 
                                  (df['runs_scored'] / df['balls_faced']) * 100, 0)
    
    # Strike Rate bonuses/penalties (min 10 balls)
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
    
    # ===== BOWLING POINTS =====
    # Wickets: +30 per wicket (CORRECTED from 25)
    df['fantasy_points'] += df['wickets'] * 30
    
    # Wicket bonuses (CORRECTED: 5-wicket bonus is 16, not 12)
    wicket_bonus = np.where(df['wickets'] >= 5, 16,
                   np.where(df['wickets'] == 4, 8,
                   np.where(df['wickets'] == 3, 4, 0)))
    df['fantasy_points'] += wicket_bonus
    
    # Maiden overs: +12 per maiden (NEW)
    df['overs_bowled'] = df['balls_bowled'] / 6.0
    maiden_overs = ((df['balls_bowled'] >= 6) & (df['runs_conceded'] == 0)).astype(int)
    df['fantasy_points'] += maiden_overs * 12
    
    # Calculate economy rate
    df['economy_rate'] = np.where(df['overs_bowled'] > 0,
                                   df['runs_conceded'] / df['overs_bowled'], 0)
    
    # Economy Rate bonuses/penalties (min 2 overs) (NEW)
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
    
    # ===== FIELDING POINTS =====
    # Catches: +8 per catch
    df['fantasy_points'] += df['catches'] * 8
    
    # 3-catch bonus: +4 (NEW)
    catch_bonus = np.where(df['catches'] >= 3, 4, 0)
    df['fantasy_points'] += catch_bonus
    
    # Stumpings: +12 per stumping (if stumpings column exists)
    if 'stumpings' in df.columns:
        df['fantasy_points'] += df['stumpings'] * 12
    
    # Run outs: +6 per run out (conservative estimate, as we don't have direct/indirect breakdown)
    if 'run_outs' in df.columns:
        df['fantasy_points'] += df['run_outs'] * 6
    
    logging.info("Fantasy points calculation complete using OFFICIAL Dream11 T20 rules.")
    logging.info(f"Average fantasy points per player-match: {df['fantasy_points'].mean():.2f}")
    return df


def create_rolling_features(df):
    """
    Enhanced rolling features with weighted averages and volatility metrics.
    Maintains backward compatibility while adding advanced features.
    """
    logging.info("Creating enhanced rolling features...")
    
    # Ensure data is sorted
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['player', 'date']).reset_index(drop=True)
    
    grouped = df.groupby('player')
    
    # Define rolling windows (same as original)
    windows = [3, 5, 10]
    stats_to_roll = ['fantasy_points', 'runs_scored', 'wickets']
    
    # Original rolling features (KEEP SAME)
    for stat in stats_to_roll:
        for w in windows:
            df[f'roll_{stat}_{w}'] = grouped[stat].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            ).fillna(0)
    
    # NEW: Add rolling standard deviation for consistency
    for stat in stats_to_roll:
        for w in windows:
            df[f'roll_{stat}_{w}_std'] = grouped[stat].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).std()
            ).fillna(0)
    
    # NEW: Weighted rolling average (recent matches weighted more)
    def weighted_rolling_mean(series, window=5):
        weights = np.arange(1, window + 1)
        return series.shift(1).rolling(window, min_periods=1).apply(
            lambda x: np.average(x, weights=weights[:len(x)]) if len(x) > 0 else 0,
            raw=True
        )
    
    df['weighted_fp_5'] = grouped['fantasy_points'].transform(
        lambda x: weighted_rolling_mean(x, 5)
    ).fillna(0)
    
    # NEW: Form trend (last 3 vs previous 3)
    df['form_trend'] = (
        df['roll_fantasy_points_3'] - 
        grouped['fantasy_points'].transform(
            lambda x: x.shift(4).rolling(3, min_periods=1).mean()
        )
    ).fillna(0)
    
    # NEW: Consistency score
    df['consistency_score'] = np.where(
        df['roll_fantasy_points_5'] > 0,
        1 / (1 + df['roll_fantasy_points_5_std'] / (df['roll_fantasy_points_5'] + 0.1)),
        0
    )
    
    logging.info("Enhanced rolling features created successfully.")
    return df


def create_venue_features(df):
    """
    Engineers venue-based features (KEPT FROM ORIGINAL).
    """
    logging.info("Creating venue features...")
    
    # Calculate player's average fantasy points at each venue
    venue_avg = df.groupby(['player', 'venue'])['fantasy_points'].mean().reset_index()
    venue_avg = venue_avg.rename(columns={'fantasy_points': 'venue_avg_fp'})
    
    # Merge back
    df = pd.merge(df, venue_avg, on=['player', 'venue'], how='left')
    
    # Prevent data leakage
    df['venue_avg_fp'] = df.groupby(['player', 'venue'])['venue_avg_fp'].transform(
        lambda x: x.shift(1)
    ).fillna(0)
    
    logging.info("Venue features created successfully.")
    return df


def create_contextual_features(df):
    """
    NEW: Create match frequency and experience features.
    """
    logging.info("Creating contextual features...")
    
    df = df.sort_values(by=['player', 'date']).reset_index(drop=True)
    
    # Match count (experience)
    df['match_count'] = df.groupby('player').cumcount() + 1
    
    # Days since last match
    df['days_since_last_match'] = df.groupby('player')['date'].diff().dt.days.fillna(7)
    
    logging.info("Contextual features created successfully.")
    return df
def create_categorical_encodings(df):
    """
    Create encoded versions of categorical variables for model training.
    """
    logging.info("Creating categorical encodings...")
    
    # Encode venue
    if 'venue' in df.columns:
        df['venue_encoded'] = df['venue'].astype('category').cat.codes
    else:
        df['venue_encoded'] = 0
    
    # Encode team
    if 'team' in df.columns:
        df['team_encoded'] = df['team'].astype('category').cat.codes
    else:
        df['team_encoded'] = 0
    
    # Encode city if available
    if 'city' in df.columns:
        df['city_encoded'] = df['city'].astype('category').cat.codes
    else:
        df['city_encoded'] = 0
    
    logging.info("Categorical encodings created successfully.")
    return df

if __name__ == '__main__':
    INTERIM_DATA_PATH = 'data/interim/player_match_stats.parquet'
    PROCESSED_DATA_PATH = 'data/processed/final_model_data.parquet'
    
    interim_path = Path(INTERIM_DATA_PATH)
    if not interim_path.exists():
        logging.error(f"{INTERIM_DATA_PATH} not found. Please run data_preprocessing.py first.")
    else:
        logging.info("Loading interim data...")
        df = pd.read_parquet(interim_path)
        
        # VERIFY T20 ONLY
        if 'match_type' in df.columns:
            non_t20 = df[df['match_type'] != 'T20']
            if len(non_t20) > 0:
                logging.warning(f"Found {len(non_t20)} non-T20 records! Filtering them out...")
                df = df[df['match_type'] == 'T20'].copy()
            
            logging.info(f"Processing {len(df)} records from T20 matches only")
            logging.info(f"Unique T20 matches: {df['match_id'].nunique()}")
        else:
            logging.warning("No match_type column found. Assuming all data is T20.")
        
        logging.info(f"Initial data shape: {df.shape}")
        
        # Apply all transformations
        df = calculate_fantasy_points(df)
        df = create_rolling_features(df)
        df = create_venue_features(df)
        df = create_contextual_features(df)
        df = create_categorical_encodings(df)
        
        # Save
        processed_path = Path(PROCESSED_DATA_PATH)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed_path, index=False)
        
        logging.info(f"Final data shape: {df.shape}")
        logging.info(f"Feature engineering complete. Dataset saved to {processed_path}")
