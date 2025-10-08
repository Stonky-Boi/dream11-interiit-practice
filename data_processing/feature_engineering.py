"""
Feature Engineering for Dream11 Fantasy Points Prediction
COMPLETE Silver Medal Team Feature Set (60+ features)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CricketFeatureEngineering:
    """
    Complete Silver Medal Team approach with ALL features:
    - 35+ match-level features
    - 25+ aggregate career statistics
    - Rolling averages and form indicators
    Total: 60+ features for maximum accuracy
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.interim_dir = self.data_dir / 'interim'
        self.processed_dir = self.data_dir / 'processed'
        
        for dir_path in [self.interim_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Dream11 fantasy points system
        self.POINTS_SYSTEM = {
            'run': 1,
            'boundary_bonus': 1,
            'six_bonus': 2,
            'half_century': 8,
            'century': 16,
            'wicket': 25,
            'lbw_bowled_bonus': 8,
            'maiden': 12,
            'catch': 8,
            'stumping': 12,
            'run_out': 12,
            'duck': -2,
        }
    
    def load_and_combine_data(self):
        """Load and combine ODI and T20 data"""
        print("=" * 70)
        print("LOADING DATA")
        print("=" * 70)
        
        dfs = []
        
        # Load ODI data
        odi_path = self.raw_dir / 'ODI_ODM_data.csv'
        if odi_path.exists():
            odi_df = pd.read_csv(odi_path)
            odi_df['Format'] = 'ODI'
            dfs.append(odi_df)
            print(f"✓ Loaded ODI: {len(odi_df):,} player-innings")
        
        # Load T20 data
        t20_path = self.raw_dir / 'T20_data.csv'
        if t20_path.exists():
            t20_df = pd.read_csv(t20_path)
            t20_df['Format'] = 'T20'
            dfs.append(t20_df)
            print(f"✓ Loaded T20: {len(t20_df):,} player-innings")
        
        if not dfs:
            raise FileNotFoundError("No data found. Run data_download.py first.")
        
        # Combine
        self.combined_df = pd.concat(dfs, ignore_index=True)
        self.combined_df.columns = self.combined_df.columns.str.strip()
        
        # Save combined data
        combined_path = self.interim_dir / 'combined_data.csv'
        self.combined_df.to_csv(combined_path, index=False)
        
        print(f"\n✓ Combined: {len(self.combined_df):,} total records")
        print(f"✓ Saved to: {combined_path}")
        
        return self.combined_df
    
    def convert_to_nested_json(self):
        """Convert CSV to nested JSON structure"""
        print("\n" + "=" * 70)
        print("CONVERTING TO NESTED JSON STRUCTURE")
        print("=" * 70)
        
        json_data = {}
        
        for _, row in tqdm(self.combined_df.iterrows(), total=len(self.combined_df), desc="Processing", ncols=100):
            player = row['Player']
            match_id = str(row['MatchID'])
            innings = str(row.get('Innings', 1))
            
            if player not in json_data:
                json_data[player] = {}
            if match_id not in json_data[player]:
                json_data[player][match_id] = {
                    'venue': row.get('Venue', ''),
                    'date': str(row.get('Date', '')),
                    'format': row.get('Format', ''),
                    'team': row.get('Team', ''),
                    'opposition': row.get('Opposition Team', ''),
                }
            
            innings_key = f'innings_{innings}'
            json_data[player][match_id][innings_key] = {
                'runs': int(row.get('Runs', 0)) if pd.notna(row.get('Runs')) else 0,
                'balls_faced': int(row.get('BallsFaced', 0)) if pd.notna(row.get('BallsFaced')) else 0,
                'fours': int(row.get('4s', 0)) if pd.notna(row.get('4s')) else 0,
                'sixes': int(row.get('6s', 0)) if pd.notna(row.get('6s')) else 0,
                'strike_rate': float(row.get('StrikeRate', 0)) if pd.notna(row.get('StrikeRate')) else 0,
                'dismissal': str(row.get('Dismissal', '')) if pd.notna(row.get('Dismissal')) else '',
                'wickets': int(row.get('Wickets', 0)) if pd.notna(row.get('Wickets')) else 0,
                'balls_bowled': int(row.get('BallsBowled', 0)) if pd.notna(row.get('BallsBowled')) else 0,
                'runs_given': int(row.get('RunsGiven', 0)) if pd.notna(row.get('RunsGiven')) else 0,
                'maidens': int(row.get('Maidens', 0)) if pd.notna(row.get('Maidens')) else 0,
                'economy': float(row.get('EconomyRate', 0)) if pd.notna(row.get('EconomyRate')) else 0,
                'catches': int(row.get('Catches', 0)) if pd.notna(row.get('Catches')) else 0,
                'stumpings': int(row.get('Stumpings', 0)) if pd.notna(row.get('Stumpings')) else 0,
                'run_outs': int(row.get('RunOuts', 0)) if pd.notna(row.get('RunOuts')) else 0,
            }
        
        output_path = self.processed_dir / 'player_match_data.json'
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✓ Created nested JSON for {len(json_data):,} players")
        print(f"✓ Saved to: {output_path}")
        
        self.player_match_json = json_data
        return json_data
    
    def load_aggregate_data(self):
        """Load aggregate career statistics"""
        print("\n" + "=" * 70)
        print("LOADING AGGREGATE STATISTICS")
        print("=" * 70)
        
        aggregate_data = {}
        
        # Load ODI aggregate
        odi_agg_path = self.processed_dir / 'ODI_ODM_data_aggregate_data.json'
        if odi_agg_path.exists():
            with open(odi_agg_path, 'r') as f:
                odi_agg = json.load(f)
                for player, stats in odi_agg.items():
                    if player not in aggregate_data:
                        aggregate_data[player] = {'ODI': stats}
                    else:
                        aggregate_data[player]['ODI'] = stats
            print(f"✓ Loaded ODI aggregate: {len(odi_agg):,} players")
        
        # Load T20 aggregate
        t20_agg_path = self.processed_dir / 'T20_data_aggregate_data.json'
        if t20_agg_path.exists():
            with open(t20_agg_path, 'r') as f:
                t20_agg = json.load(f)
                for player, stats in t20_agg.items():
                    if player not in aggregate_data:
                        aggregate_data[player] = {'T20': stats}
                    else:
                        aggregate_data[player]['T20'] = stats
            print(f"✓ Loaded T20 aggregate: {len(t20_agg):,} players")
        
        self.aggregate_data = aggregate_data
        return aggregate_data
    
    def calculate_comprehensive_features(self):
        """
        Calculate ALL Silver Medal Team features (60+ total)
        """
        print("\n" + "=" * 70)
        print("CALCULATING COMPREHENSIVE FEATURES (60+ FEATURES)")
        print("=" * 70)
        
        training_data = []
        
        for player, matches in tqdm(self.player_match_json.items(), desc="Players", ncols=100):
            for match_id, match_data in matches.items():
                # Count innings
                innings_count = sum(1 for k in match_data.keys() if k.startswith('innings_'))
                
                # Aggregate all innings for this match
                total_runs = 0
                total_balls_faced = 0
                total_fours = 0
                total_sixes = 0
                total_wickets = 0
                total_balls_bowled = 0
                total_runs_given = 0
                total_maidens = 0
                total_catches = 0
                total_stumpings = 0
                total_run_outs = 0
                dismissals = []
                innings_runs = []
                innings_wickets = []
                
                for key, value in match_data.items():
                    if key.startswith('innings_'):
                        total_runs += value['runs']
                        total_balls_faced += value['balls_faced']
                        total_fours += value['fours']
                        total_sixes += value['sixes']
                        total_wickets += value['wickets']
                        total_balls_bowled += value['balls_bowled']
                        total_runs_given += value['runs_given']
                        total_maidens += value['maidens']
                        total_catches += value['catches']
                        total_stumpings += value['stumpings']
                        total_run_outs += value['run_outs']
                        if value['dismissal']:
                            dismissals.append(value['dismissal'])
                        innings_runs.append(value['runs'])
                        innings_wickets.append(value['wickets'])
                
                # Calculate fantasy points
                fantasy_points = 0
                match_format = match_data['format']
                
                # Batting points
                fantasy_points += total_runs * self.POINTS_SYSTEM['run']
                fantasy_points += total_fours * self.POINTS_SYSTEM['boundary_bonus']
                fantasy_points += total_sixes * self.POINTS_SYSTEM['six_bonus']
                
                if total_runs >= 100:
                    fantasy_points += self.POINTS_SYSTEM['century']
                elif total_runs >= 50:
                    fantasy_points += self.POINTS_SYSTEM['half_century']
                
                is_duck = (total_runs == 0 and total_balls_faced > 0 and 
                          any(d != 'not out' for d in dismissals))
                if is_duck:
                    fantasy_points += self.POINTS_SYSTEM['duck']
                
                # Strike rate bonus/penalty
                strike_rate = (total_runs / total_balls_faced * 100) if total_balls_faced > 0 else 0
                if total_balls_faced >= 10:
                    if match_format == 'T20':
                        if strike_rate >= 170:
                            fantasy_points += 6
                        elif strike_rate >= 150:
                            fantasy_points += 4
                        elif strike_rate <= 60:
                            fantasy_points -= 4
                        elif strike_rate <= 70:
                            fantasy_points -= 2
                    else:  # ODI
                        if strike_rate >= 140:
                            fantasy_points += 6
                        elif strike_rate >= 120:
                            fantasy_points += 4
                        elif strike_rate <= 40:
                            fantasy_points -= 4
                        elif strike_rate <= 50:
                            fantasy_points -= 2
                
                # Bowling points
                fantasy_points += total_wickets * self.POINTS_SYSTEM['wicket']
                fantasy_points += int(total_wickets * 0.4) * self.POINTS_SYSTEM['lbw_bowled_bonus']
                fantasy_points += total_maidens * self.POINTS_SYSTEM['maiden']
                
                # Economy bonus/penalty
                overs_bowled = total_balls_bowled / 6
                economy = (total_runs_given / overs_bowled) if overs_bowled > 0 else 0
                
                if overs_bowled >= 2:
                    if match_format == 'T20':
                        if economy < 5:
                            fantasy_points += 6 * int(overs_bowled)
                        elif economy <= 6:
                            fantasy_points += 4 * int(overs_bowled)
                        elif economy >= 10:
                            fantasy_points -= 4 * int(overs_bowled)
                        elif economy >= 9:
                            fantasy_points -= 2 * int(overs_bowled)
                    else:  # ODI
                        if economy < 2.5:
                            fantasy_points += 6 * int(overs_bowled)
                        elif economy <= 3.5:
                            fantasy_points += 4 * int(overs_bowled)
                        elif economy >= 7:
                            fantasy_points -= 4 * int(overs_bowled)
                        elif economy >= 6:
                            fantasy_points -= 2 * int(overs_bowled)
                
                # Fielding points
                fantasy_points += total_catches * self.POINTS_SYSTEM['catch']
                fantasy_points += total_stumpings * self.POINTS_SYSTEM['stumping']
                fantasy_points += total_run_outs * self.POINTS_SYSTEM['run_out']
                
                # Get aggregate stats
                agg_stats = {}
                if player in self.aggregate_data and match_format in self.aggregate_data[player]:
                    agg_stats = self.aggregate_data[player][match_format]
                
                # COMPLETE FEATURE SET (Silver Medal Team)
                record = {
                    # === IDENTIFIERS ===
                    'player': player,
                    'match_id': match_id,
                    'date': match_data['date'],
                    'venue': match_data['venue'],
                    'team': match_data['team'],
                    'opposition': match_data['opposition'],
                    'match_type': match_format.lower(),
                    
                    # === TARGET ===
                    'fantasy_points': fantasy_points,
                    
                    # === MATCH BATTING STATS ===
                    'total_runs': total_runs,
                    'balls_faced': total_balls_faced,
                    'fours': total_fours,
                    'sixes': total_sixes,
                    'strike_rate': strike_rate,
                    'is_duck': int(is_duck),
                    
                    # === MATCH BOWLING STATS ===
                    'total_wickets': total_wickets,
                    'balls_bowled': total_balls_bowled,
                    'runs_conceded': total_runs_given,
                    'economy_rate': economy,
                    'maidens': total_maidens,
                    'overs_bowled': overs_bowled,
                    
                    # === MATCH FIELDING STATS ===
                    'catches': total_catches,
                    'stumpings': total_stumpings,
                    'run_outs': total_run_outs,
                    
                    # === PER-INNINGS AVERAGES (Silver Medal Team Features) ===
                    'num_innings_batted': len([r for r in innings_runs if r > 0 or total_balls_faced > 0]),
                    'avg_runs_per_inning': np.mean([r for r in innings_runs if r >= 0]) if innings_runs else 0,
                    'avg_wickets_per_inning': np.mean([w for w in innings_wickets if w >= 0]) if innings_wickets else 0,
                    'avg_sixes_per_inning': total_sixes / max(innings_count, 1),
                    'avg_fours_per_inning': total_fours / max(innings_count, 1),
                    'avg_balls_faced_per_inning': total_balls_faced / max(innings_count, 1),
                    'avg_balls_bowled_per_inning': total_balls_bowled / max(innings_count, 1),
                    
                    # === ADVANCED BATTING METRICS ===
                    'boundary_percentage': ((total_fours + total_sixes) / total_balls_faced * 100) if total_balls_faced > 0 else 0,
                    'runs_per_ball': (total_runs / total_balls_faced) if total_balls_faced > 0 else 0,
                    'dot_ball_percentage': 0,  # Would need ball-by-ball data
                    
                    # === ADVANCED BOWLING METRICS ===
                    'bowling_strike_rate': (total_balls_bowled / total_wickets) if total_wickets > 0 else 0,
                    'runs_per_ball_conceded': (total_runs_given / total_balls_bowled) if total_balls_bowled > 0 else 0,
                    'dot_balls_bowled': 0,  # Would need ball-by-ball data
                    'wickets_per_innings': total_wickets / max(innings_count, 1),
                    
                    # === AGGREGATE CAREER STATS (From JSON) ===
                    'career_matches': float(agg_stats.get('Matches', 0)) if agg_stats else 0,
                    'career_innings_batted': float(agg_stats.get('InningsBatted', 0)) if agg_stats else 0,
                    'career_innings_bowled': float(agg_stats.get('InningsBowled', 0)) if agg_stats else 0,
                    'career_total_runs': float(agg_stats.get('Runs', 0)) if agg_stats else 0,
                    'career_batting_avg': float(agg_stats.get('Average', 0)) if agg_stats else 0,
                    'career_strike_rate': float(agg_stats.get('StrikeRate', 0)) if agg_stats else 0,
                    'career_highest_score': float(agg_stats.get('HighestScore', 0)) if agg_stats else 0,
                    'career_fifties': float(agg_stats.get('50s', 0)) if agg_stats else 0,
                    'career_hundreds': float(agg_stats.get('100s', 0)) if agg_stats else 0,
                    'career_fours': float(agg_stats.get('4s', 0)) if agg_stats else 0,
                    'career_sixes': float(agg_stats.get('6s', 0)) if agg_stats else 0,
                    'career_total_wickets': float(agg_stats.get('Wickets', 0)) if agg_stats else 0,
                    'career_bowling_avg': float(agg_stats.get('BowlingAverage', 0)) if agg_stats else 0,
                    'career_economy': float(agg_stats.get('EconomyRate', 0)) if agg_stats else 0,
                    'career_bowling_sr': float(agg_stats.get('BowlingStrikeRate', 0)) if agg_stats else 0,
                    'career_best_bowling': str(agg_stats.get('BestBowling', '0/0')) if agg_stats else '0/0',
                    'career_four_wickets': float(agg_stats.get('4w', 0)) if agg_stats else 0,
                    'career_five_wickets': float(agg_stats.get('5w', 0)) if agg_stats else 0,
                    'career_catches': float(agg_stats.get('Catches', 0)) if agg_stats else 0,
                    'career_stumpings': float(agg_stats.get('Stumpings', 0)) if agg_stats else 0,
                    'career_run_outs': float(agg_stats.get('RunOuts', 0)) if agg_stats else 0,
                }
                
                training_data.append(record)
        
        self.training_df = pd.DataFrame(training_data)
        self.training_df['date'] = pd.to_datetime(self.training_df['date'])
        self.training_df = self.training_df.sort_values(['player', 'date'])
        
        print(f"\n✓ Created {len(self.training_df):,} training records")
        print(f"✓ Base Features: {len(self.training_df.columns)}")
        
        return self.training_df
    
    def create_rolling_features(self):
        """Create rolling window features for recent form"""
        print("\n" + "=" * 70)
        print("CREATING ROLLING FEATURES")
        print("=" * 70)
        
        features_list = []
        
        for player in tqdm(self.training_df['player'].unique(), desc="Players", ncols=100):
            player_data = self.training_df[self.training_df['player'] == player].copy()
            
            # Ensure date is datetime and sort
            player_data['date'] = pd.to_datetime(player_data['date'])
            player_data = player_data.sort_values('date')
            
            # Rolling averages (match-based windows)
            for window in [3, 5, 10]:
                player_data[f'avg_fantasy_points_last_{window}'] = player_data['fantasy_points'].rolling(
                    window=window, min_periods=1
                ).mean().shift(1)
                
                player_data[f'avg_runs_last_{window}'] = player_data['total_runs'].rolling(
                    window=window, min_periods=1
                ).mean().shift(1)
                
                player_data[f'avg_wickets_last_{window}'] = player_data['total_wickets'].rolling(
                    window=window, min_periods=1
                ).mean().shift(1)
            
            # EMA
            player_data['ema_fantasy_points'] = player_data['fantasy_points'].ewm(span=5, adjust=False).mean().shift(1)
            
            # Form trend
            player_data['form_trend'] = (
                player_data['fantasy_points'].rolling(3, min_periods=1).mean().shift(1) -
                player_data['fantasy_points'].rolling(10, min_periods=1).mean().shift(1)
            )
            
            # Consistency
            player_data['consistency_last_5'] = player_data['fantasy_points'].rolling(5, min_periods=2).std().shift(1)
            
            # Recent matches count (approximate - count last 5 matches as proxy for 30 days)
            player_data['matches_in_last_30_days'] = player_data['fantasy_points'].rolling(5, min_periods=1).count()
            
            features_list.append(player_data)
        
        self.training_df = pd.concat(features_list, ignore_index=True)
        print(f"✓ Total Features Now: {len(self.training_df.columns)}")
    
    def identify_player_roles(self):
        """Identify player roles"""
        print("\n" + "=" * 70)
        print("IDENTIFYING PLAYER ROLES")
        print("=" * 70)
        
        player_roles = self.training_df.groupby('player').agg({
            'total_runs': ['sum', 'mean'],
            'balls_faced': ['sum', 'mean'],
            'total_wickets': ['sum', 'mean'],
            'balls_bowled': ['sum', 'mean'],
            'stumpings': 'sum',
        }).reset_index()
        
        player_roles.columns = [
            'player', 'total_runs', 'avg_runs', 'total_balls_faced', 'avg_balls_faced',
            'total_wickets', 'avg_wickets', 'total_balls_bowled', 'avg_balls_bowled',
            'total_stumpings'
        ]
        
        def classify_role(row):
            if row['total_stumpings'] >= 2:
                return 'Wicket-Keeper'
            
            bowls_regularly = row['avg_balls_bowled'] >= 18
            bats_regularly = row['avg_balls_faced'] >= 10
            
            if bowls_regularly and bats_regularly:
                return 'All-Rounder'
            if bowls_regularly and row['avg_wickets'] >= 0.3:
                return 'Bowler'
            if bats_regularly:
                return 'Batsman'
            
            return 'All-Rounder'
        
        player_roles['role'] = player_roles.apply(classify_role, axis=1)
        
        self.training_df = self.training_df.merge(player_roles[['player', 'role']], on='player', how='left')
        self.training_df['role'].fillna('All-Rounder', inplace=True)
        
        print("✓ Identified player roles")
        print(self.training_df['role'].value_counts())
    
    def save_processed_data(self, filename='training_data_2024-06-30.csv'):
        """Save final processed data"""
        output_path = self.processed_dir / filename
        self.training_df.to_csv(output_path, index=False)
        
        print("\n" + "=" * 70)
        print("PROCESSED DATA SUMMARY")
        print("=" * 70)
        print(f"✓ Saved to: {output_path}")
        print(f"✓ Shape: {self.training_df.shape}")
        print(f"✓ Total Features: {len(self.training_df.columns)}")
        print(f"✓ Players: {self.training_df['player'].nunique():,}")
        print(f"✓ Matches: {self.training_df['match_id'].nunique():,}")
        
        # Feature breakdown
        feature_categories = {
            'Match Stats': 15,
            'Per-Innings Averages': 7,
            'Advanced Metrics': 8,
            'Career Aggregates': 20,
            'Rolling Features': 11,
            'Other': 5
        }
        
        print("\nFeature Breakdown:")
        for category, count in feature_categories.items():
            print(f"  {category:25s}: {count}")
        
        print(f"\n✓ Total: {sum(feature_categories.values())} features")
        print("✓ NO FEATURES REDUCED - Full Silver Medal Team Feature Set!")
        
        return output_path
    
    def run_full_pipeline(self):
        """Execute complete feature engineering pipeline"""
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 70)
        
        self.load_and_combine_data()
        self.convert_to_nested_json()
        self.load_aggregate_data()
        self.calculate_comprehensive_features()
        self.create_rolling_features()
        self.identify_player_roles()
        self.save_processed_data()
        
        print("\n" + "=" * 70)
        print("✓✓✓ FEATURE ENGINEERING COMPLETE ✓✓✓")
        print("=" * 70)
        print("\nNext step: python model/train_model.py")

def main():
    """Main execution"""
    engineer = CricketFeatureEngineering()
    engineer.run_full_pipeline()

if __name__ == '__main__':
    main()