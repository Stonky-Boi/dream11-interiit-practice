"""
Data Download Script for Dream11 Inter-IIT Project
Downloads cricket data from Cricsheet (ODIs and T20s only)
"""

import requests
import zipfile
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CricketDataDownloader:
    """Download and process cricket data from Cricsheet"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw' / 'cricksheet_data'
        self.interim_dir = self.data_dir / 'interim'
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        
        # Cricsheet URLs - ODIs and T20s (both male and female combined)
        self.data_urls = {
            'odi': 'https://cricsheet.org/downloads/odis_json.zip',      # 3,019 matches
            't20': 'https://cricsheet.org/downloads/t20s_json.zip',      # 4,673 matches
        }
    
    def download_data(self, formats=['odi', 't20']):
        """Download cricket data from Cricsheet"""
        print("=" * 70)
        print("DOWNLOADING CRICKET DATA FROM CRICSHEET")
        print("=" * 70)
        print("\nDownloading ODIs and T20 Internationals (Male + Female)")
        print("This will provide maximum data coverage for training")
        print("-" * 70)
        
        for format_type in formats:
            if format_type not in self.data_urls:
                print(f"Skipping unknown format: {format_type}")
                continue
                
            url = self.data_urls[format_type]
            zip_path = self.raw_dir / f'{format_type}_json.zip'
            extract_path = self.raw_dir / format_type
            
            print(f"\n[{format_type.upper()}] Downloading from Cricsheet...")
            
            try:
                # Download with progress bar
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(zip_path, 'wb') as f, tqdm(
                    desc=f"{format_type.upper()} Download",
                    total=total_size, 
                    unit='B', 
                    unit_scale=True,
                    unit_divisor=1024
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                
                print(f"[{format_type.upper()}] ✓ Download complete. Extracting...")
                
                # Extract
                extract_path.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                
                # Count extracted files
                json_files = list(extract_path.glob('*.json'))
                print(f"[{format_type.upper()}] ✓ Extracted {len(json_files)} match files")
                
                # Clean up zip file to save space
                zip_path.unlink()
                
            except Exception as e:
                print(f"[{format_type.upper()}] ✗ Error downloading: {str(e)}")
                print(f"Please check internet connection or try again later")
    
    def parse_match_json(self, json_path):
        """Parse a single match JSON file from Cricsheet"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                match_data = json.load(f)
            
            # Extract basic match info
            info = match_data.get('info', {})
            
            # Get teams
            teams = info.get('teams', [])
            team1 = teams[0] if len(teams) > 0 else None
            team2 = teams[1] if len(teams) > 1 else None
            
            # Get outcome
            outcome = info.get('outcome', {})
            winner = outcome.get('winner', None)
            
            match_info = {
                'match_id': json_path.stem,
                'date': info.get('dates', [None])[0],
                'team1': team1,
                'team2': team2,
                'venue': info.get('venue', ''),
                'city': info.get('city', ''),
                'toss_winner': info.get('toss', {}).get('winner', ''),
                'toss_decision': info.get('toss', {}).get('decision', ''),
                'match_type': info.get('match_type', ''),
                'match_type_number': info.get('match_type_number', ''),
                'gender': info.get('gender', 'male'),
                'overs': info.get('overs', 50),
                'winner': winner,
                'competition': info.get('event', {}).get('name', '') if isinstance(info.get('event'), dict) else ''
            }
            
            # Get player registrations (squads)
            registry = info.get('registry', {}).get('people', {})
            players = info.get('players', {})
            
            # Get innings data
            innings_data = match_data.get('innings', [])
            
            return match_info, players, innings_data, registry
            
        except Exception as e:
            return None, None, None, None
    
    def process_all_matches(self, formats=['odi', 't20']):
        """Process ALL downloaded match JSONs into structured data (no date filtering)"""
        print("\n" + "=" * 70)
        print("PROCESSING ALL MATCH DATA (NO DATE FILTERING)")
        print("=" * 70)
        print("Processing entire dataset - will split by date later")
        print("-" * 70)
        
        all_matches = []
        all_ball_by_ball = []
        
        skipped_no_date = 0
        processed_count = 0
        
        for format_type in formats:
            format_dir = self.raw_dir / format_type
            if not format_dir.exists():
                print(f"[{format_type.upper()}] ✗ Format directory not found: {format_dir}")
                continue
            
            json_files = list(format_dir.glob('*.json'))
            print(f"\n[{format_type.upper()}] Found {len(json_files)} match files")
            print(f"[{format_type.upper()}] Processing...")
            
            for json_file in tqdm(json_files, desc=f"{format_type.upper()}", ncols=100):
                match_info, players, innings_data, registry = self.parse_match_json(json_file)
                
                if match_info is None:
                    continue
                
                # Skip if no date
                if not match_info['date']:
                    skipped_no_date += 1
                    continue
                
                match_info['format'] = format_type
                all_matches.append(match_info)
                processed_count += 1
                
                # Process ball-by-ball data with team assignment
                for innings in innings_data:
                    team = innings.get('team', '')
                    innings_num = innings.get('innings', 0)
                    
                    for over in innings.get('overs', []):
                        over_num = over.get('over', 0)
                        for delivery in over.get('deliveries', []):
                            # Extract wicket information
                            wickets = delivery.get('wickets', [])
                            wicket_type = ''
                            wicket_player = ''
                            if wickets:
                                wicket_type = wickets[0].get('kind', '')
                                wicket_player = wickets[0].get('player_out', '')
                            
                            ball_data = {
                                'match_id': match_info['match_id'],
                                'innings': innings_num,
                                'batting_team': team,
                                'bowling_team': match_info['team2'] if team == match_info['team1'] else match_info['team1'],
                                'over': over_num,
                                'ball': delivery.get('ball', 0),
                                'batter': delivery.get('batter', ''),
                                'bowler': delivery.get('bowler', ''),
                                'non_striker': delivery.get('non_striker', ''),
                                'runs_batter': delivery.get('runs', {}).get('batter', 0),
                                'runs_extras': delivery.get('runs', {}).get('extras', 0),
                                'runs_total': delivery.get('runs', {}).get('total', 0),
                                'wickets': len(wickets),
                                'wicket_type': wicket_type,
                                'wicket_player': wicket_player
                            }
                            all_ball_by_ball.append(ball_data)
        
        # Create DataFrames
        matches_df = pd.DataFrame(all_matches)
        balls_df = pd.DataFrame(all_ball_by_ball)
        
        # Convert dates
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        
        # Save ALL data (interim)
        matches_csv = self.interim_dir / 'matches_all.csv'
        balls_csv = self.interim_dir / 'balls_all.csv'
        
        matches_df.to_csv(matches_csv, index=False)
        balls_df.to_csv(balls_csv, index=False)
        
        # Print summary
        print("\n" + "=" * 70)
        print("PROCESSING SUMMARY")
        print("=" * 70)
        print(f"✓ Total matches processed:    {processed_count:,}")
        print(f"✓ Total deliveries:           {len(balls_df):,}")
        print(f"✓ Unique players:             {len(set(balls_df['batter']) | set(balls_df['bowler'])):,}")
        print(f"  Skipped (no date):          {skipped_no_date}")
        print("-" * 70)
        
        if len(matches_df) > 0:
            print(f"Date range: {matches_df['date'].min().date()} to {matches_df['date'].max().date()}")
            print(f"Formats: {matches_df['format'].value_counts().to_dict()}")
            print(f"Gender: {matches_df['gender'].value_counts().to_dict()}")
        
        print(f"\n✓ Saved complete dataset to:")
        print(f"  • {matches_csv}")
        print(f"  • {balls_csv}")
        print("=" * 70)
        
        return matches_df, balls_df
    
    def split_by_date(self, cutoff_date='2024-06-30'):
        """Split data into training and test sets based on date"""
        print("\n" + "=" * 70)
        print(f"SPLITTING DATA BY DATE (CUTOFF: {cutoff_date})")
        print("=" * 70)
        
        # Load complete data
        matches_df = pd.read_csv(self.interim_dir / 'matches_all.csv')
        balls_df = pd.read_csv(self.interim_dir / 'balls_all.csv')
        
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        cutoff = pd.to_datetime(cutoff_date)
        
        # Split matches
        train_matches = matches_df[matches_df['date'] <= cutoff]
        test_matches = matches_df[matches_df['date'] > cutoff]
        
        # Split balls based on match_id
        train_match_ids = set(train_matches['match_id'])
        test_match_ids = set(test_matches['match_id'])
        
        train_balls = balls_df[balls_df['match_id'].isin(train_match_ids)]
        test_balls = balls_df[balls_df['match_id'].isin(test_match_ids)]
        
        # Save training data
        train_matches.to_csv(self.interim_dir / 'matches_train.csv', index=False)
        train_balls.to_csv(self.interim_dir / 'balls_train.csv', index=False)
        
        # Save test data
        test_dir = self.data_dir / 'out_of_sample_data'
        test_dir.mkdir(parents=True, exist_ok=True)
        test_matches.to_csv(test_dir / 'matches_test.csv', index=False)
        test_balls.to_csv(test_dir / 'balls_test.csv', index=False)
        
        print(f"\n✓ TRAINING DATA (up to {cutoff_date}):")
        print(f"  • Matches: {len(train_matches):,}")
        print(f"  • Deliveries: {len(train_balls):,}")
        print(f"  • Date range: {train_matches['date'].min().date()} to {train_matches['date'].max().date()}")
        
        print(f"\n✓ TEST DATA (after {cutoff_date}):")
        print(f"  • Matches: {len(test_matches):,}")
        print(f"  • Deliveries: {len(test_balls):,}")
        if len(test_matches) > 0:
            print(f"  • Date range: {test_matches['date'].min().date()} to {test_matches['date'].max().date()}")
        else:
            print(f"  • No test data available")
        
        print(f"\n✓ Training data saved to: {self.interim_dir}/")
        print(f"✓ Test data saved to: {test_dir}/")
        print("=" * 70)
        
        return train_matches, test_matches
    
    def validate_data_split(self, cutoff_date='2024-06-30'):
        """Validate that training/test split is correct"""
        print("\n" + "=" * 70)
        print("DATA VALIDATION - CHECKING SPLIT")
        print("=" * 70)
        
        try:
            # Check training data
            train_matches = pd.read_csv(self.interim_dir / 'matches_train.csv')
            train_matches['date'] = pd.to_datetime(train_matches['date'])
            
            cutoff = pd.to_datetime(cutoff_date)
            
            # Check for any matches after cutoff in training
            after_cutoff = train_matches[train_matches['date'] > cutoff]
            
            if len(after_cutoff) > 0:
                print(f"⚠️  WARNING: Found {len(after_cutoff)} matches after {cutoff_date} in TRAINING data!")
                print("⚠️  This will lead to DISQUALIFICATION!")
                return False
            else:
                print(f"✓ VALIDATION PASSED: No training data after {cutoff_date}")
                print(f"✓ Training data ends: {train_matches['date'].max().date()}")
            
            # Check test data
            test_path = self.data_dir / 'out_of_sample_data' / 'matches_test.csv'
            if test_path.exists():
                test_matches = pd.read_csv(test_path)
                test_matches['date'] = pd.to_datetime(test_matches['date'])
                
                before_cutoff = test_matches[test_matches['date'] <= cutoff]
                
                if len(before_cutoff) > 0:
                    print(f"\n⚠️  WARNING: Found {len(before_cutoff)} matches before {cutoff_date} in TEST data!")
                    print("⚠️  Test data should only contain matches after training cutoff!")
                else:
                    print(f"\n✓ Test data validation passed")
                    print(f"✓ Test data starts: {test_matches['date'].min().date()}")
            
            print("=" * 70)
            return True
                
        except Exception as e:
            print(f"✗ Validation error: {str(e)}")
            return False

def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("DREAM11 INTER-IIT DATA PIPELINE")
    print("=" * 70)
    print("Downloading ODIs and T20s (Male + Female) from Cricsheet")
    print("Processing entire dataset, then splitting by 2024-06-30")
    print("=" * 70 + "\n")
    
    downloader = CricketDataDownloader()
    
    # Step 1: Download data
    print("\n[STEP 1/4] Downloading data...")
    downloader.download_data(formats=['odi', 't20'])
    
    # Step 2: Process ALL matches (no filtering)
    print("\n[STEP 2/4] Processing all matches...")
    matches_df, balls_df = downloader.process_all_matches(formats=['odi', 't20'])
    
    # Step 3: Split by date
    print("\n[STEP 3/4] Splitting data by cutoff date...")
    train_matches, test_matches = downloader.split_by_date(cutoff_date='2024-06-30')
    
    # Step 4: Validate split
    print("\n[STEP 4/4] Validating data split...")
    validation_passed = downloader.validate_data_split(cutoff_date='2024-06-30')
    
    print("\n" + "=" * 70)
    if validation_passed:
        print("✓✓✓ DATA DOWNLOAD COMPLETE AND VALIDATED ✓✓✓")
    else:
        print("⚠️⚠️⚠️ DATA VALIDATION FAILED - CHECK WARNINGS ⚠️⚠️⚠️")
    print("=" * 70)
    
    print("\nNext steps:")
    print("  1. Run: python data_processing/feature_engineering.py")
    print("  2. Run: python model/train_model.py")
    print("  3. Run: streamlit run main_app.py")

if __name__ == '__main__':
    main()