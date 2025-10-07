"""
Feature Engineering for Dream11 Fantasy Points Prediction
Comprehensive feature creation from cricket match data (ODIs and T20s)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CricketFeatureEngineering:
    """Feature engineering for cricket fantasy points prediction"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.interim_dir = self.data_dir / 'interim'
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Fantasy points system (Dream11)
        self.POINTS_SYSTEM = {
            'run': 1,
            'boundary_bonus': 1,  # 4 runs
            'six_bonus': 2,  # 6 runs
            'half_century': 8,
            'century': 16,
            'wicket': 25,
            'lbw_bowled': 8,  # Bonus for LBW/Bowled
            'maiden': 12,
            'catch': 8,
            'stumping': 12,
            'run_out': 12,
            'duck': -2,  # Batting duck
            # Economy rate bonuses (ODI)
            'economy_rate_below_2.5': 6,
            'economy_rate_2.5_3.5': 4,
            'economy_rate_7_8': -2,
            'economy_rate_above_8': -4,
            # Economy rate bonuses (T20)
            'economy_rate_below_5': 6,
            'economy_rate_5_6': 4,
            'economy_rate_9_10': -2,
            'economy_rate_above_10': -4,
            # Strike rate bonuses
            'strike_rate_above_170': 6,
            'strike_rate_150_170': 4,
            'strike_rate_60_70': -2,
            'strike_rate_below_60': -4,
        }
    
    def load_raw_data(self, data_type='train'):
        """Load raw processed data (train or test)"""
        print("=" * 70)
        print(f"LOADING {data_type.upper()} DATA")
        print("=" * 70)
        
        if data_type == 'train':
            matches_path = self.interim_dir / 'matches_train.csv'
            balls_path = self.interim_dir / 'balls_train.csv'
        elif data_type == 'test':
            test_dir = self.data_dir / 'out_of_sample_data'
            matches_path = test_dir / 'matches_test.csv'
            balls_path = test_dir / 'balls_test.csv'
        else:
            raise ValueError("data_type must be 'train' or 'test'")
        
        if not matches_path.exists() or not balls_path.exists():
            raise FileNotFoundError(f"Data files not found. Please run data_download.py first.")
        
        self.matches_df = pd.read_csv(matches_path)
        self.balls_df = pd.read_csv(balls_path)
        
        # Convert dates
        self.matches_df['date'] = pd.to_datetime(self.matches_df['date'])
        
        print(f"✓ Loaded {len(self.matches_df):,} matches")
        print(f"✓ Loaded {len(self.balls_df):,} deliveries")
        print(f"✓ Date range: {self.matches_df['date'].min()} to {self.matches_df['date'].max()}")
        print(f"✓ Formats: {self.matches_df['format'].value_counts().to_dict()}")
    
    def calculate_fantasy_points(self):
        """Calculate fantasy points for each player in each match"""
        print("\n" + "=" * 70)
        print("CALCULATING FANTASY POINTS")
        print("=" * 70)
        
        player_match_stats = []
        
        for match_id in tqdm(self.matches_df['match_id'].unique(), desc="Processing matches", ncols=100):
            match_balls = self.balls_df[self.balls_df['match_id'] == match_id]
            match_info = self.matches_df[self.matches_df['match_id'] == match_id].iloc[0]
            
            # Get all players in the match
            batters = set(match_balls['batter'].unique())
            bowlers = set(match_balls['bowler'].unique())
            players = batters | bowlers
            
            for player in players:
                stats = self._calculate_player_match_stats(player, match_balls, match_info)
                if stats:
                    player_match_stats.append(stats)
        
        self.player_match_df = pd.DataFrame(player_match_stats)
        print(f"\n✓ Calculated fantasy points for {len(self.player_match_df):,} player-match combinations")
        print(f"✓ Unique players: {self.player_match_df['player'].nunique():,}")
        
        return self.player_match_df
    
    def _calculate_player_match_stats(self, player, match_balls, match_info):
        """Calculate detailed stats and fantasy points for a player in a match"""
        
        # Determine player's team
        batting_team = match_balls[match_balls['batter'] == player]['batting_team'].mode()
        bowling_team = match_balls[match_balls['bowler'] == player]['bowling_team'].mode()
        
        if len(batting_team) > 0:
            player_team = batting_team.iloc[0]
        elif len(bowling_team) > 0:
            player_team = bowling_team.iloc[0]
        else:
            return None
        
        opposition_team = match_info['team2'] if player_team == match_info['team1'] else match_info['team1']
        
        # Batting stats
        batting_balls = match_balls[match_balls['batter'] == player]
        runs = int(batting_balls['runs_batter'].sum())
        balls_faced = len(batting_balls)
        fours = len(batting_balls[batting_balls['runs_batter'] == 4])
        sixes = len(batting_balls[batting_balls['runs_batter'] == 6])
        
        # Check for duck (dismissed for 0)
        wicket_balls = match_balls[match_balls['wicket_player'] == player]
        is_duck = (runs == 0 and len(wicket_balls) > 0)
        
        strike_rate = (runs / balls_faced * 100) if balls_faced > 0 else 0
        
        # Bowling stats
        bowling_balls = match_balls[match_balls['bowler'] == player]
        wickets = int(bowling_balls['wickets'].sum())
        runs_conceded = int(bowling_balls['runs_total'].sum())
        balls_bowled = len(bowling_balls)
        overs_bowled = balls_bowled / 6
        
        # Wicket types (bonus for LBW/Bowled)
        lbw_bowled_wickets = len(bowling_balls[
            bowling_balls['wicket_type'].isin(['lbw', 'bowled'])
        ])
        
        # Calculate maidens (overs with 0 runs)
        maidens = 0
        if balls_bowled > 0:
            # Group by over and count maidens
            bowling_by_over = bowling_balls.groupby('over')['runs_total'].sum()
            maidens = (bowling_by_over == 0).sum()
        
        economy_rate = (runs_conceded / overs_bowled) if overs_bowled > 0 else 0
        
        # Fielding stats (simplified - Cricsheet has limited fielding data)
        # We can extract catches from wicket_type
        catches = len(match_balls[
            (match_balls['wicket_type'] == 'caught') & 
            (match_balls['wickets'] > 0)
        ])
        
        stumpings = len(match_balls[
            (match_balls['wicket_type'] == 'stumped') & 
            (match_balls['wickets'] > 0)
        ])
        
        run_outs = len(match_balls[
            (match_balls['wicket_type'] == 'run out') & 
            (match_balls['wickets'] > 0)
        ])
        
        # Calculate fantasy points
        fantasy_points = 0
        
        # Batting points
        fantasy_points += runs * self.POINTS_SYSTEM['run']
        fantasy_points += fours * self.POINTS_SYSTEM['boundary_bonus']
        fantasy_points += sixes * self.POINTS_SYSTEM['six_bonus']
        
        if runs >= 100:
            fantasy_points += self.POINTS_SYSTEM['century']
        elif runs >= 50:
            fantasy_points += self.POINTS_SYSTEM['half_century']
        
        if is_duck:
            fantasy_points += self.POINTS_SYSTEM['duck']
        
        # Strike rate bonus/penalty (min 10 balls for T20, 20 for ODI)
        min_balls = 10 if match_info['format'] == 't20' else 20
        if balls_faced >= min_balls:
            if strike_rate >= 170:
                fantasy_points += self.POINTS_SYSTEM['strike_rate_above_170']
            elif strike_rate >= 150:
                fantasy_points += self.POINTS_SYSTEM['strike_rate_150_170']
            elif strike_rate <= 60:
                fantasy_points += self.POINTS_SYSTEM['strike_rate_below_60']
            elif strike_rate <= 70:
                fantasy_points += self.POINTS_SYSTEM['strike_rate_60_70']
        
        # Bowling points
        fantasy_points += wickets * self.POINTS_SYSTEM['wicket']
        fantasy_points += lbw_bowled_wickets * self.POINTS_SYSTEM['lbw_bowled']
        fantasy_points += maidens * self.POINTS_SYSTEM['maiden']
        
        # Economy rate bonus/penalty (min 2 overs)
        if overs_bowled >= 2:
            if match_info['format'] == 't20':
                if economy_rate < 5:
                    fantasy_points += self.POINTS_SYSTEM['economy_rate_below_5'] * int(overs_bowled)
                elif economy_rate <= 6:
                    fantasy_points += self.POINTS_SYSTEM['economy_rate_5_6'] * int(overs_bowled)
                elif economy_rate >= 10:
                    fantasy_points += self.POINTS_SYSTEM['economy_rate_above_10'] * int(overs_bowled)
                elif economy_rate >= 9:
                    fantasy_points += self.POINTS_SYSTEM['economy_rate_9_10'] * int(overs_bowled)
            else:  # ODI
                if economy_rate < 2.5:
                    fantasy_points += self.POINTS_SYSTEM['economy_rate_below_2.5'] * int(overs_bowled)
                elif economy_rate <= 3.5:
                    fantasy_points += self.POINTS_SYSTEM['economy_rate_2.5_3.5'] * int(overs_bowled)
                elif economy_rate >= 8:
                    fantasy_points += self.POINTS_SYSTEM['economy_rate_above_8'] * int(overs_bowled)
                elif economy_rate >= 7:
                    fantasy_points += self.POINTS_SYSTEM['economy_rate_7_8'] * int(overs_bowled)
        
        # Fielding points
        fantasy_points += catches * self.POINTS_SYSTEM['catch']
        fantasy_points += stumpings * self.POINTS_SYSTEM['stumping']
        fantasy_points += run_outs * self.POINTS_SYSTEM['run_out']
        
        return {
            'match_id': match_info['match_id'],
            'date': match_info['date'],
            'player': player,
            'team': player_team,
            'venue': match_info['venue'],
            'city': match_info['city'],
            'opposition': opposition_team,
            'match_type': match_info['format'],
            'gender': match_info['gender'],
            'toss_winner': match_info['toss_winner'],
            'toss_decision': match_info['toss_decision'],
            # Batting
            'runs': runs,
            'balls_faced': balls_faced,
            'fours': fours,
            'sixes': sixes,
            'strike_rate': strike_rate,
            'is_duck': int(is_duck),
            # Bowling
            'wickets': wickets,
            'runs_conceded': runs_conceded,
            'overs_bowled': overs_bowled,
            'economy_rate': economy_rate,
            'maidens': maidens,
            # Fielding
            'catches': catches,
            'stumpings': stumpings,
            'run_outs': run_outs,
            # Fantasy points
            'fantasy_points': fantasy_points
        }
    
    def create_historical_features(self):
        """Create rolling averages and historical performance features"""
        print("\n" + "=" * 70)
        print("CREATING HISTORICAL FEATURES")
        print("=" * 70)
        
        # Sort by date
        self.player_match_df = self.player_match_df.sort_values(['player', 'date'])
        
        features_list = []
        
        print("Computing rolling averages for each player...")
        for player in tqdm(self.player_match_df['player'].unique(), desc="Players", ncols=100):
            player_data = self.player_match_df[self.player_match_df['player'] == player].copy()
            
            # Rolling averages (last N matches)
            for window in [3, 5, 10]:
                player_data[f'avg_fantasy_points_last_{window}'] = player_data['fantasy_points'].rolling(
                    window=window, min_periods=1
                ).mean().shift(1)
                
                player_data[f'avg_runs_last_{window}'] = player_data['runs'].rolling(
                    window=window, min_periods=1
                ).mean().shift(1)
                
                player_data[f'avg_wickets_last_{window}'] = player_data['wickets'].rolling(
                    window=window, min_periods=1
                ).mean().shift(1)
                
                player_data[f'avg_strike_rate_last_{window}'] = player_data['strike_rate'].rolling(
                    window=window, min_periods=1
                ).mean().shift(1)
                
                player_data[f'avg_economy_last_{window}'] = player_data['economy_rate'].rolling(
                    window=window, min_periods=1
                ).mean().shift(1)
            
            # Exponential moving average (recent form emphasis)
            player_data['ema_fantasy_points'] = player_data['fantasy_points'].ewm(
                span=5, adjust=False
            ).mean().shift(1)
            
            # Career statistics
            player_data['career_matches'] = range(1, len(player_data) + 1)
            player_data['career_avg_fantasy_points'] = player_data['fantasy_points'].expanding().mean().shift(1)
            player_data['career_avg_runs'] = player_data['runs'].expanding().mean().shift(1)
            player_data['career_avg_wickets'] = player_data['wickets'].expanding().mean().shift(1)
            player_data['career_avg_strike_rate'] = player_data['strike_rate'].expanding().mean().shift(1)
            player_data['career_avg_economy'] = player_data['economy_rate'].expanding().mean().shift(1)
            
            # Recent form trend (momentum)
            player_data['form_trend'] = player_data['fantasy_points'].rolling(
                window=3, min_periods=1
            ).mean().shift(1) - player_data['fantasy_points'].rolling(
                window=10, min_periods=1
            ).mean().shift(1)
            
            # Consistency (standard deviation of recent performances)
            player_data['consistency_last_5'] = player_data['fantasy_points'].rolling(
                window=5, min_periods=2
            ).std().shift(1)
            
            features_list.append(player_data)
        
        self.player_match_df = pd.concat(features_list, ignore_index=True)
        print("✓ Created historical features")
    
    def create_venue_features(self):
        """Create venue-specific features"""
        print("\n" + "=" * 70)
        print("CREATING VENUE FEATURES")
        print("=" * 70)
        
        # Sort by date to avoid data leakage
        self.player_match_df = self.player_match_df.sort_values(['player', 'date'])
        
        venue_features_list = []
        
        for player in tqdm(self.player_match_df['player'].unique(), desc="Players", ncols=100):
            player_data = self.player_match_df[self.player_match_df['player'] == player].copy()
            
            # Calculate cumulative venue stats (shift to avoid leakage)
            venue_stats = []
            for idx, row in player_data.iterrows():
                # Get all previous matches at this venue
                prev_at_venue = player_data[
                    (player_data['venue'] == row['venue']) & 
                    (player_data['date'] < row['date'])
                ]
                
                if len(prev_at_venue) > 0:
                    venue_avg_fp = prev_at_venue['fantasy_points'].mean()
                    venue_matches = len(prev_at_venue)
                    venue_avg_runs = prev_at_venue['runs'].mean()
                    venue_avg_wickets = prev_at_venue['wickets'].mean()
                else:
                    venue_avg_fp = np.nan
                    venue_matches = 0
                    venue_avg_runs = np.nan
                    venue_avg_wickets = np.nan
                
                venue_stats.append({
                    'venue_avg_fantasy_points': venue_avg_fp,
                    'venue_matches': venue_matches,
                    'venue_avg_runs': venue_avg_runs,
                    'venue_avg_wickets': venue_avg_wickets
                })
            
            venue_df = pd.DataFrame(venue_stats, index=player_data.index)
            player_data = pd.concat([player_data, venue_df], axis=1)
            venue_features_list.append(player_data)
        
        self.player_match_df = pd.concat(venue_features_list, ignore_index=True)
        print("✓ Created venue features")
    
    def create_opposition_features(self):
        """Create opposition-specific features"""
        print("\n" + "=" * 70)
        print("CREATING OPPOSITION FEATURES")
        print("=" * 70)
        
        opposition_features_list = []
        
        for player in tqdm(self.player_match_df['player'].unique(), desc="Players", ncols=100):
            player_data = self.player_match_df[self.player_match_df['player'] == player].copy()
            
            # Calculate cumulative opposition stats
            opp_stats = []
            for idx, row in player_data.iterrows():
                prev_vs_opp = player_data[
                    (player_data['opposition'] == row['opposition']) & 
                    (player_data['date'] < row['date'])
                ]
                
                if len(prev_vs_opp) > 0:
                    opp_avg_fp = prev_vs_opp['fantasy_points'].mean()
                    opp_matches = len(prev_vs_opp)
                    opp_avg_runs = prev_vs_opp['runs'].mean()
                    opp_avg_wickets = prev_vs_opp['wickets'].mean()
                else:
                    opp_avg_fp = np.nan
                    opp_matches = 0
                    opp_avg_runs = np.nan
                    opp_avg_wickets = np.nan
                
                opp_stats.append({
                    'opp_avg_fantasy_points': opp_avg_fp,
                    'opp_matches': opp_matches,
                    'opp_avg_runs': opp_avg_runs,
                    'opp_avg_wickets': opp_avg_wickets
                })
            
            opp_df = pd.DataFrame(opp_stats, index=player_data.index)
            player_data = pd.concat([player_data, opp_df], axis=1)
            opposition_features_list.append(player_data)
        
        self.player_match_df = pd.concat(opposition_features_list, ignore_index=True)
        print("✓ Created opposition features")
    
    def identify_player_roles(self):
        """Identify player roles based on historical performance"""
        print("\n" + "=" * 70)
        print("IDENTIFYING PLAYER ROLES")
        print("=" * 70)
        
        # Aggregate career statistics per player
        player_roles = self.player_match_df.groupby('player').agg({
            'runs': ['sum', 'mean', 'count'],
            'balls_faced': ['sum', 'mean'],
            'wickets': ['sum', 'mean'],
            'overs_bowled': ['sum', 'mean'],
            'runs_conceded': 'sum',
            'stumpings': 'sum',  # Total stumpings in career
            'catches': 'sum'
        }).reset_index()
        
        # Flatten column names
        player_roles.columns = [
            'player', 
            'total_runs', 'avg_runs', 'matches',
            'total_balls_faced', 'avg_balls_faced',
            'total_wickets', 'avg_wickets',
            'total_overs_bowled', 'avg_overs_bowled',
            'total_runs_conceded',
            'total_stumpings',
            'total_catches'
        ]
        
        def classify_role(row):
            """
            Classify player role based on career statistics
            
            Logic:
            - Wicket-Keeper: Has stumpings (typically 2+) in career
            - All-Rounder: Significant batting AND bowling (both contribute)
            - Bowler: Bowls regularly (avg 3+ overs per match)
            - Batsman: Bats regularly without much bowling
            """
            
            # Check if genuine wicket-keeper (stumpings are rare and specific to WKs)
            if row['total_stumpings'] >= 2:
                return 'Wicket-Keeper'
            
            # Calculate bowling and batting contributions
            bowls_regularly = row['avg_overs_bowled'] >= 3.0
            bats_regularly = row['avg_balls_faced'] >= 10.0
            
            # Check if significant contributor in both departments
            significant_batting = row['avg_runs'] >= 15.0
            significant_bowling = row['avg_wickets'] >= 0.5
            
            # All-rounder: Both bat and bowl with significant contribution
            if bowls_regularly and bats_regularly:
                if significant_batting and significant_bowling:
                    return 'All-Rounder'
                # If bowls and bats but not significant in one
                elif significant_bowling:
                    return 'All-Rounder'  # Bowling all-rounder
                elif significant_batting:
                    return 'All-Rounder'  # Batting all-rounder
                else:
                    return 'All-Rounder'  # General all-rounder
            
            # Pure bowler: Bowls regularly, doesn't bat much
            if bowls_regularly and row['avg_wickets'] >= 0.3:
                return 'Bowler'
            
            # Pure batsman: Bats regularly, doesn't bowl much
            if bats_regularly and not bowls_regularly:
                return 'Batsman'
            
            # Default classification based on primary contribution
            if row['total_runs'] > row['total_wickets'] * 20:  # Runs weighted more
                return 'Batsman'
            elif row['total_wickets'] > 0 and row['avg_overs_bowled'] >= 1.5:
                return 'Bowler'
            else:
                return 'Batsman'  # Default to batsman if unclear
        
        player_roles['role'] = player_roles.apply(classify_role, axis=1)
        
        # Merge back to main dataframe
        self.player_match_df = self.player_match_df.merge(
            player_roles[['player', 'role']], on='player', how='left'
        )
        
        # Fill any missing roles with 'All-Rounder' as safe default
        self.player_match_df['role'].fillna('All-Rounder', inplace=True)
        
        print("✓ Identified player roles")
        print("\nRole distribution:")
        role_dist = self.player_match_df['role'].value_counts()
        for role, count in role_dist.items():
            percentage = (count / len(self.player_match_df)) * 100
            print(f"  {role:15s}: {count:6,} ({percentage:5.1f}%)")
        
        print("\nUnique players by role:")
        player_role_dist = self.player_match_df.groupby('player')['role'].first().value_counts()
        for role, count in player_role_dist.items():
            print(f"  {role:15s}: {count:4,} players")
    
    def create_contextual_features(self):
        """Create match context features"""
        print("\n" + "=" * 70)
        print("CREATING CONTEXTUAL FEATURES")
        print("=" * 70)
        
        # Days since last match
        self.player_match_df['days_since_last_match'] = self.player_match_df.groupby('player')['date'].diff().dt.days
        self.player_match_df['days_since_last_match'].fillna(30, inplace=True)
        
        # Temporal features
        self.player_match_df['month'] = pd.to_datetime(self.player_match_df['date']).dt.month
        self.player_match_df['year'] = pd.to_datetime(self.player_match_df['date']).dt.year
        self.player_match_df['day_of_year'] = pd.to_datetime(self.player_match_df['date']).dt.dayofyear
        
        # Toss impact features
        self.player_match_df['won_toss'] = (
            self.player_match_df['team'] == self.player_match_df['toss_winner']
        ).astype(int)
        
        self.player_match_df['toss_bat'] = (
            self.player_match_df['toss_decision'] == 'bat'
        ).astype(int)
        
        print("✓ Created contextual features")
    
    def save_processed_data(self, filename='training_data_2024-06-30.csv'):
        """Save processed feature data"""
        output_path = self.processed_dir / filename
        self.player_match_df.to_csv(output_path, index=False)
        
        print("\n" + "=" * 70)
        print("SAVING PROCESSED DATA")
        print("=" * 70)
        print(f"✓ Saved to: {output_path}")
        print(f"✓ Shape: {self.player_match_df.shape}")
        print(f"✓ Features: {len(self.player_match_df.columns)}")
        print(f"✓ Players: {self.player_match_df['player'].nunique():,}")
        print(f"✓ Matches: {self.player_match_df['match_id'].nunique():,}")
        
        # Show sample statistics
        print("\nFantasy Points Statistics:")
        print(self.player_match_df['fantasy_points'].describe())
        
        return output_path
    
    def run_full_pipeline(self, data_type='train', output_filename=None):
        """Execute complete feature engineering pipeline"""
        print("\n" + "=" * 70)
        print(f"FEATURE ENGINEERING PIPELINE ({data_type.upper()})")
        print("=" * 70)
        
        self.load_raw_data(data_type=data_type)
        self.calculate_fantasy_points()
        self.create_historical_features()
        self.create_venue_features()
        self.create_opposition_features()
        self.identify_player_roles()
        self.create_contextual_features()
        
        if output_filename is None:
            if data_type == 'train':
                output_filename = 'training_data_2024-06-30.csv'
            else:
                output_filename = 'test_data_after_2024-06-30.csv'
        
        output_path = self.save_processed_data(output_filename)
        
        print("\n" + "=" * 70)
        print(f"✓✓✓ FEATURE ENGINEERING COMPLETE ({data_type.upper()}) ✓✓✓")
        print("=" * 70)
        
        if data_type == 'train':
            print("\nNext step: python model/train_model.py")
        else:
            print("\nTest data ready for evaluation!")
        
        return output_path

def main():
    """Main execution"""
    engineer = CricketFeatureEngineering()
    
    # Process training data
    print("\n" + "=" * 70)
    print("PROCESSING TRAINING DATA")
    print("=" * 70)
    engineer.run_full_pipeline(data_type='train')
    
    # Process test data if available
    test_path = Path('data/out_of_sample_data/matches_test.csv')
    if test_path.exists():
        print("\n\n" + "=" * 70)
        print("PROCESSING TEST DATA")
        print("=" * 70)
        engineer_test = CricketFeatureEngineering()
        engineer_test.run_full_pipeline(data_type='test')
    else:
        print("\n\n⚠️  No test data found. Run data download first to get test matches.")

if __name__ == '__main__':
    main()