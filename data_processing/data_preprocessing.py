import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_match_json(file_path):
    """
    Parses a single Cricsheet JSON file - T20 MATCHES ONLY with GENDER field.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    info = data.get('info', {})
    
    # CRITICAL: Filter for T20 format only
    match_type = info.get('match_type', '').lower()
    match_type_number = info.get('match_type_number', None)
    
    is_t20 = (match_type_number == 3 or 
              't20' in match_type or 
              'twenty20' in match_type.replace(' ', ''))
    
    if not is_t20:
        return []
    
    # CRITICAL: Extract gender
    gender = info.get('gender', 'male').lower()
    if gender not in ['male', 'female']:
        gender = 'male'
    
    match_id = file_path.stem
    
    player_to_team = {}
    if 'players' in info:
        for team, players in info['players'].items():
            for player in players:
                player_to_team[player] = team
    
    player_stats = defaultdict(lambda: defaultdict(int))
    
    all_players = set()
    if 'players' in info:
        for team_players in info['players'].values():
            all_players.update(team_players)
    
    for player in all_players:
        player_stats[player]
    
    for inning in data.get('innings', []):
        for over in inning.get('overs', []):
            for delivery in over.get('deliveries', []):
                batter = delivery['batter']
                bowler = delivery['bowler']
                runs = delivery['runs']['batter']
                total_runs = delivery['runs']['total']
                extras = delivery['runs'].get('extras', 0)
                
                player_stats[batter]['runs_scored'] += runs
                player_stats[batter]['balls_faced'] += 1
                
                if runs == 4:
                    player_stats[batter]['fours'] += 1
                elif runs == 6:
                    player_stats[batter]['sixes'] += 1
                
                player_stats[bowler]['balls_bowled'] += 1
                player_stats[bowler]['runs_conceded'] += total_runs
                
                if runs == 0 and extras == 0:
                    player_stats[bowler]['dots_bowled'] += 1
                
                if 'wickets' in delivery:
                    for wicket in delivery['wickets']:
                        kind = wicket.get('kind', '')
                        
                        if kind not in ['run out', 'retired hurt', 'retired out', 'obstructing the field']:
                            player_stats[bowler]['wickets'] += 1
                        
                        if kind == 'caught':
                            for fielder_info in wicket.get('fielders', []):
                                fielder = fielder_info.get('name')
                                if fielder:
                                    player_stats[fielder]['catches'] += 1
                        
                        elif kind == 'stumped':
                            for fielder_info in wicket.get('fielders', []):
                                fielder = fielder_info.get('name')
                                if fielder:
                                    player_stats[fielder]['stumpings'] += 1
                        
                        elif kind == 'run out':
                            for fielder_info in wicket.get('fielders', []):
                                fielder = fielder_info.get('name')
                                if fielder:
                                    player_stats[fielder]['run_outs'] += 1
    
    records = []
    for player, stats in player_stats.items():
        record = {
            'match_id': match_id,
            'player': player,
            'team': player_to_team.get(player, 'Unknown'),
            'date': info.get('dates', [''])[0],
            'venue': info.get('venue', 'Unknown'),
            'city': info.get('city', 'Unknown'),
            'match_type': 'T20',
            'gender': gender,  # NEW: Gender field
            'runs_scored': stats['runs_scored'],
            'balls_faced': stats['balls_faced'],
            'fours': stats['fours'],
            'sixes': stats['sixes'],
            'wickets': stats['wickets'],
            'balls_bowled': stats['balls_bowled'],
            'runs_conceded': stats['runs_conceded'],
            'dots_bowled': stats['dots_bowled'],
            'catches': stats['catches'],
            'stumpings': stats['stumpings'],
            'run_outs': stats['run_outs']
        }
        records.append(record)
    
    return records


def process_all_matches(raw_data_dir, output_path):
    """
    Process all T20 matches with gender tracking - SAVE AS CSV.
    """
    raw_path = Path(raw_data_dir)
    
    if not raw_path.exists():
        logging.error(f"Directory does not exist: {raw_data_dir}")
        return
    
    json_files = list(raw_path.glob('*.json'))
    
    if not json_files:
        logging.error(f"No JSON files found in {raw_data_dir}")
        return
    
    logging.info(f"Found {len(json_files)} JSON files to process.")
    
    all_records = []
    t20_matches = 0
    skipped_matches = 0
    male_matches = 0
    female_matches = 0
    
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            records = parse_match_json(json_file)
            
            if records:
                all_records.extend(records)
                t20_matches += 1
                
                if records[0]['gender'] == 'male':
                    male_matches += 1
                else:
                    female_matches += 1
            else:
                skipped_matches += 1
                
        except Exception as e:
            logging.warning(f"Failed to process {json_file.name}: {e}")
            skipped_matches += 1
    
    if not all_records:
        logging.error("No T20 records extracted!")
        return
    
    df = pd.DataFrame(all_records)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # CHANGED: Save as CSV instead of Parquet
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    logging.info(f"\n{'='*80}")
    logging.info("PROCESSING SUMMARY")
    logging.info(f"{'='*80}")
    logging.info(f"Total files: {len(json_files)}")
    logging.info(f"T20 matches: {t20_matches} (Male: {male_matches}, Female: {female_matches})")
    logging.info(f"Skipped (non-T20): {skipped_matches}")
    logging.info(f"Player-match records: {len(df)}")
    logging.info(f"Unique players: {df['player'].nunique()}")
    logging.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logging.info(f"\nGender distribution:")
    logging.info(f"  Male: {len(df[df['gender'] == 'male'])} records")
    logging.info(f"  Female: {len(df[df['gender'] == 'female'])} records")
    logging.info(f"Data saved to: {output_path}")
    logging.info(f"{'='*80}")


if __name__ == '__main__':
    RAW_DATA_DIR = 'data/raw/cricsheet_data'
    INTERIM_DATA_PATH = 'data/interim/player_match_stats.csv'  # CHANGED to .csv
    
    logging.info(f"Starting T20-only data preprocessing with gender tracking...")
    process_all_matches(RAW_DATA_DIR, INTERIM_DATA_PATH)
