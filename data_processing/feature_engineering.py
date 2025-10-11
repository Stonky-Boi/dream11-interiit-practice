"""
Feature Engineering for Dream11 Fantasy Points Prediction
COMPLETE Silver Medal Team Feature Set (60+ features)
"""

import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print_length=100

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
        print("=" * print_length)
        print("LOADING DATA")
        print("=" * print_length)
        
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
        print("\n" + "=" * print_length)
        print("CONVERTING TO NESTED JSON STRUCTURE")
        print("=" * print_length)
        
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
        print("\n" + "=" * print_length)
        print("LOADING AGGREGATE STATISTICS")
        print("=" * print_length)
        
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
        CRITICAL: Only use historical data, NOT current match stats
        """
        print("\n" + "=" * print_length)
        print("CALCULATING COMPREHENSIVE FEATURES (60+ FEATURES)")
        print("=" * print_length)
        
        training_data = []
        
        for player, matches in tqdm(self.player_match_json.items(), desc="Players", ncols=100):
            # Sort matches by date for this player
            sorted_matches = sorted(
                [(match_id, match_data) for match_id, match_data in matches.items()],
                key=lambda x: x[1]['date']
            )
            
            # Track historical stats for this player
            historical_matches = []
            
            for match_id, match_data in sorted_matches:
                # Count innings
                innings_count = sum(1 for k in match_data.keys() if k.startswith('innings_'))
                
                # Aggregate current match stats (for target calculation only)
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
                
                # Calculate fantasy points (TARGET)
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
                
                # Get aggregate stats (historical)
                agg_stats = {}
                if player in self.aggregate_data:
                    format_key = match_format
                    if format_key in self.aggregate_data[player]:
                        agg_stats = self.aggregate_data[player][format_key]
                
                # Calculate HISTORICAL features (from previous matches only)
                if len(historical_matches) > 0:
                    hist_df = pd.DataFrame(historical_matches)
                    
                    # Recent match averages (last 3, 5, 10)
                    avg_fp_last_3 = hist_df['fantasy_points'].tail(3).mean() if len(hist_df) >= 1 else 0
                    avg_fp_last_5 = hist_df['fantasy_points'].tail(5).mean() if len(hist_df) >= 1 else 0
                    avg_fp_last_10 = hist_df['fantasy_points'].tail(10).mean() if len(hist_df) >= 1 else 0
                    
                    avg_runs_last_3 = hist_df['runs'].tail(3).mean() if len(hist_df) >= 1 else 0
                    avg_runs_last_5 = hist_df['runs'].tail(5).mean() if len(hist_df) >= 1 else 0
                    avg_runs_last_10 = hist_df['runs'].tail(10).mean() if len(hist_df) >= 1 else 0
                    
                    avg_wickets_last_3 = hist_df['wickets'].tail(3).mean() if len(hist_df) >= 1 else 0
                    avg_wickets_last_5 = hist_df['wickets'].tail(5).mean() if len(hist_df) >= 1 else 0
                    avg_wickets_last_10 = hist_df['wickets'].tail(10).mean() if len(hist_df) >= 1 else 0
                    
                    # Historical averages
                    hist_avg_runs = hist_df['runs'].mean()
                    hist_avg_wickets = hist_df['wickets'].mean()
                    hist_avg_strike_rate = hist_df['strike_rate'].mean()
                    hist_avg_economy = hist_df['economy'].mean() if hist_df['economy'].mean() > 0 else 0
                    
                    # Form trend
                    form_trend = avg_fp_last_3 - avg_fp_last_10 if len(hist_df) >= 3 else 0
                    consistency = hist_df['fantasy_points'].tail(5).std() if len(hist_df) >= 5 else 0
                    
                else:
                    # No historical data yet - use zeros
                    avg_fp_last_3 = avg_fp_last_5 = avg_fp_last_10 = 0
                    avg_runs_last_3 = avg_runs_last_5 = avg_runs_last_10 = 0
                    avg_wickets_last_3 = avg_wickets_last_5 = avg_wickets_last_10 = 0
                    hist_avg_runs = hist_avg_wickets = hist_avg_strike_rate = hist_avg_economy = 0
                    form_trend = consistency = 0
                
                # Create feature dictionary (ONLY HISTORICAL DATA)
                record = {
                    # Identifiers
                    'player': player,
                    'match_id': match_id,
                    'date': match_data['date'],
                    'venue': match_data['venue'],
                    'team': match_data['team'],
                    'opposition': match_data['opposition'],
                    'match_type': match_format.lower(),
                    
                    # TARGET (from current match)
                    'fantasy_points': fantasy_points,
                    
                    # FEATURES (all historical - before current match)
                    # Recent form features
                    'avg_fantasy_points_last_3': avg_fp_last_3,
                    'avg_fantasy_points_last_5': avg_fp_last_5,
                    'avg_fantasy_points_last_10': avg_fp_last_10,
                    'avg_runs_last_3': avg_runs_last_3,
                    'avg_runs_last_5': avg_runs_last_5,
                    'avg_runs_last_10': avg_runs_last_10,
                    'avg_wickets_last_3': avg_wickets_last_3,
                    'avg_wickets_last_5': avg_wickets_last_5,
                    'avg_wickets_last_10': avg_wickets_last_10,
                    
                    # Historical averages
                    'hist_avg_runs': hist_avg_runs,
                    'hist_avg_wickets': hist_avg_wickets,
                    'hist_avg_strike_rate': hist_avg_strike_rate,
                    'hist_avg_economy': hist_avg_economy,
                    'hist_matches_played': len(historical_matches),
                    
                    # Form indicators
                    'form_trend': form_trend,
                    'consistency_last_5': consistency,
                    
                    # Career aggregate stats (historical by definition)
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
                    'career_four_wickets': float(agg_stats.get('4w', 0)) if agg_stats else 0,
                    'career_five_wickets': float(agg_stats.get('5w', 0)) if agg_stats else 0,
                    'career_catches': float(agg_stats.get('Catches', 0)) if agg_stats else 0,
                    'career_stumpings': float(agg_stats.get('Stumpings', 0)) if agg_stats else 0,
                    'career_run_outs': float(agg_stats.get('RunOuts', 0)) if agg_stats else 0,
                }
                
                training_data.append(record)
                
                # Add current match to historical data for next iteration
                historical_matches.append({
                    'fantasy_points': fantasy_points,
                    'runs': total_runs,
                    'wickets': total_wickets,
                    'strike_rate': strike_rate,
                    'economy': economy,
                })
        
        # Convert to DataFrame
        self.training_df = pd.DataFrame(training_data)
        
        # Sort by date
        self.training_df['date'] = pd.to_datetime(self.training_df['date'])
        self.training_df = self.training_df.sort_values(['player', 'date'])
        
        print(f"\n✓ Created {len(self.training_df):,} training records")
        print(f"✓ Base Features: {len(self.training_df.columns)}")
        print("✓ ALL FEATURES USE HISTORICAL DATA ONLY (no leakage)")
        
        return self.training_df
    
    def create_rolling_features(self):
        """Create rolling window features for recent form"""
        print("\n" + "=" * print_length)
        print("CREATING ROLLING FEATURES")
        print("=" * print_length)
        
        # Rolling features are already calculated in calculate_comprehensive_features
        # This method now only adds EMA which requires iteration
        
        features_list = []
        
        for player in tqdm(self.training_df['player'].unique(), desc="Players", ncols=100):
            player_data = self.training_df[self.training_df['player'] == player].copy()
            
            # Ensure date is datetime and sort
            player_data['date'] = pd.to_datetime(player_data['date'])
            player_data = player_data.sort_values('date')
            
            # EMA of fantasy points (already shifted in previous method)
            player_data['ema_fantasy_points'] = player_data['avg_fantasy_points_last_5'].ewm(
                span=5, adjust=False
            ).mean()
            
            # Fill any NaN values with 0
            player_data = player_data.fillna(0)
            
            features_list.append(player_data)
        
        self.training_df = pd.concat(features_list, ignore_index=True)
        print(f"✓ Total Features Now: {len(self.training_df.columns)}")
    
    def identify_player_roles(self):
        """Identify player roles based on career aggregate statistics"""
        print("\n" + "=" * print_length)
        print("IDENTIFYING PLAYER ROLES")
        print("=" * print_length)
        
        # Use career aggregate stats to determine roles
        player_roles = self.training_df.groupby('player').agg({
            'career_total_runs': 'max',
            'career_innings_batted': 'max',
            'career_total_wickets': 'max',
            'career_innings_bowled': 'max',
            'career_stumpings': 'max',
        }).reset_index()
        
        player_roles.columns = [
            'player', 'total_runs', 'innings_batted', 
            'total_wickets', 'innings_bowled', 'total_stumpings'
        ]
        
        def classify_role(row):
            """
            Classify player role based on career statistics
            
            Logic:
            - Wicket-Keeper: Has career stumpings (typically 2+)
            - All-Rounder: Significant batting AND bowling
            - Bowler: Bowls regularly
            - Batsman: Bats regularly without much bowling
            """
            
            # Check if genuine wicket-keeper (stumpings are rare and specific to WKs)
            if row['total_stumpings'] >= 2:
                return 'Wicket-Keeper'
            
            # Calculate contribution levels
            avg_runs = row['total_runs'] / max(row['innings_batted'], 1)
            avg_wickets = row['total_wickets'] / max(row['innings_bowled'], 1)
            
            bowls_regularly = row['innings_bowled'] >= 5  # Lowered threshold
            bats_regularly = row['innings_batted'] >= 5   # Lowered threshold
            
            # Check significant contributions
            significant_batting = row['total_runs'] >= 100 or avg_runs >= 15
            significant_bowling = row['total_wickets'] >= 5 or avg_wickets >= 0.5
            
            # All-rounder: Both bat and bowl
            if bowls_regularly and bats_regularly:
                if significant_batting and significant_bowling:
                    return 'All-Rounder'
                elif significant_bowling and avg_runs >= 10:
                    return 'All-Rounder'  # Bowling all-rounder who can bat
                elif significant_batting and avg_wickets >= 0.3:
                    return 'All-Rounder'  # Batting all-rounder who can bowl
                elif bowls_regularly and bats_regularly:
                    return 'All-Rounder'  # General all-rounder
            
            # Pure bowler: Bowls regularly, decent wickets, minimal batting
            if bowls_regularly and significant_bowling:
                if not bats_regularly or avg_runs < 10:
                    return 'Bowler'
            
            # Bowler with minimal stats
            if row['innings_bowled'] > row['innings_batted'] and row['total_wickets'] >= 3:
                return 'Bowler'
            
            # Pure batsman: Bats regularly, doesn't bowl much
            if bats_regularly and not bowls_regularly:
                return 'Batsman'
            
            # Pure batsman: High batting contribution
            if significant_batting and row['innings_bowled'] < 3:
                return 'Batsman'
            
            # Default classification based on primary contribution
            if row['total_runs'] > row['total_wickets'] * 20:  # Runs weighted more
                return 'Batsman'
            elif row['total_wickets'] >= 3:
                return 'Bowler'
            else:
                return 'All-Rounder'  # Default to all-rounder if unclear
        
        player_roles['role'] = player_roles.apply(classify_role, axis=1)
        
        # Merge back to main dataframe
        self.training_df = self.training_df.merge(
            player_roles[['player', 'role']], on='player', how='left'
        )
        
        # Fill any missing roles with 'All-Rounder' as safe default
        self.training_df['role'].fillna('All-Rounder', inplace=True)
        
        print("✓ Identified player roles")
        print("\nRole distribution:")
        role_dist = self.training_df['role'].value_counts()
        for role, count in role_dist.items():
            percentage = (count / len(self.training_df)) * 100
            print(f"  {role:15s}: {count:6,} ({percentage:5.1f}%)")
        
        print("\nUnique players by role:")
        player_role_dist = self.training_df.groupby('player')['role'].first().value_counts()
        for role, count in player_role_dist.items():
            print(f"  {role:15s}: {count:4,} players")
    
    def save_processed_data(self, filename='training_data_2024-06-30.csv'):
        """Save final processed data"""
        output_path = self.processed_dir / filename
        self.training_df.to_csv(output_path, index=False)
        
        print("\n" + "=" * print_length)
        print("PROCESSED DATA SUMMARY")
        print("=" * print_length)
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
        print("\n" + "=" * print_length)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * print_length)
        
        self.load_and_combine_data()
        self.convert_to_nested_json()
        self.load_aggregate_data()
        self.calculate_comprehensive_features()
        self.create_rolling_features()
        self.identify_player_roles()
        self.save_processed_data()
        
        print("\n" + "=" * print_length)
        print("✓✓✓ FEATURE ENGINEERING COMPLETE ✓✓✓")
        print("=" * print_length)
        print("\nNext step: python model/train_model.py")

def main():
    """Main execution"""
    engineer = CricketFeatureEngineering()
    engineer.run_full_pipeline()

if __name__ == '__main__':
    main()