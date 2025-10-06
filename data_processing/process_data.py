import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_match_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    match_id = file_path.stem
    info = data.get('info', {})
    player_to_team = {}
    if 'players' in info:
        for team, players in info['players'].items():
            for player in players:
                player_to_team[player] = team
    player_stats = defaultdict(lambda: defaultdict(int))
    all_players = set(player_to_team.keys())
    for player in all_players:
        player_stats[player]
    for inning in data.get('innings', []):
        for over in inning.get('overs', []):
            for delivery in over.get('deliveries', []):
                batter, bowler = delivery['batter'], delivery['bowler']
                runs, extras = delivery.get('runs', {}), delivery.get('extras', {})
                player_stats[batter]['runs_scored'] += runs.get('batter', 0)
                player_stats[batter]['balls_faced'] += 1
                if runs.get('batter') == 4: player_stats[batter]['fours'] += 1
                if runs.get('batter') == 6: player_stats[batter]['sixes'] += 1
                is_legal = 'wides' not in extras and 'noballs' not in extras
                if is_legal: player_stats[bowler]['balls_bowled'] += 1
                runs_from_extras = extras.get('wides', 0) + extras.get('noballs', 0)
                player_stats[bowler]['runs_conceded'] += runs.get('batter', 0) + runs_from_extras
                if 'wickets' in delivery:
                    for wicket in delivery['wickets']:
                        if wicket.get('kind') != 'run out': player_stats[bowler]['wickets'] += 1
                        if 'fielders' in wicket:
                            for fielder in wicket['fielders']:
                                if 'name' in fielder:
                                    if wicket.get('kind') == 'run out': player_stats[fielder['name']]['run_outs'] += 1
                                    elif wicket.get('kind') == 'stumped': player_stats[fielder['name']]['stumpings'] += 1
                                    else: player_stats[fielder['name']]['catches'] += 1
    records = []
    for player in all_players:
        stats = player_stats[player]
        record = {
            'match_id': match_id, 'player': player, 'team': player_to_team.get(player, 'Unknown'),
            'date': info.get('dates', [None])[0], 'venue': info.get('venue'), 'city': info.get('city'),
            **stats
        }
        records.append(record)
    return records

def process_all_matches(raw_data_dir, interim_data_path):
    raw_path, interim_path = Path(raw_data_dir), Path(interim_data_path)
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    json_files = list(raw_path.glob('*.json'))
    
    logging.info(f"Found {len(json_files)} JSON files to process.")
    all_match_data = []
    for file in tqdm(json_files, desc="Processing Cricsheet JSON files"):
        try:
            match_records = parse_match_json(file)
            if match_records: all_match_data.extend(match_records)
        except Exception as e:
            logging.error(f"Failed to process file {file.name}: {e}")

    df = pd.DataFrame(all_match_data)
    stat_cols = ['runs_scored', 'balls_faced', 'fours', 'sixes', 'balls_bowled', 'runs_conceded', 'wickets', 'catches', 'stumpings', 'run_outs']
    for col in df.columns:
        if col in stat_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    logging.info("Saving processed data to interim CSV file...")
    df.to_csv(interim_path, index=False)
    logging.info(f"Successfully saved enhanced data to {interim_path}")

if __name__ == '__main__':
    RAW_DATA_DIR = 'data/raw/cricsheet_data'
    INTERIM_DATA_PATH = 'data/interim/player_match_stats.csv'
    process_all_matches(RAW_DATA_DIR, INTERIM_DATA_PATH)