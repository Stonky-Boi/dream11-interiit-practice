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
    Parses a single Cricsheet JSON file with enhanced logic for calculating
    bowling and fielding stats based on official format documentation.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    match_id = file_path.stem
    info = data.get('info', {})
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
                runs = delivery['runs']
                extras = delivery.get('extras', {})

                # 1. Batting stats
                player_stats[batter]['runs_scored'] += runs['batter']
                player_stats[batter]['balls_faced'] += 1
                if runs['batter'] == 4:
                    player_stats[batter]['fours'] += 1
                if runs['batter'] == 6:
                    player_stats[batter]['sixes'] += 1

                # 2. Bowling stats
                is_legal_delivery = 'wides' not in extras and 'noballs' not in extras
                if is_legal_delivery:
                    player_stats[bowler]['balls_bowled'] += 1
                runs_from_extras = extras.get('wides', 0) + extras.get('noballs', 0)
                player_stats[bowler]['runs_conceded'] += runs['batter'] + runs_from_extras

                # 3. Wicket and Fielding stats
                if 'wickets' in delivery:
                    for wicket in delivery['wickets']:
                        # Bowler gets wicket credit (unless it's a run out)
                        if wicket.get('kind') != 'run out':
                             player_stats[bowler]['wickets'] += 1
                        # Fielding stats for catches, stumpings, and run outs
                        if 'fielders' in wicket:
                            for fielder in wicket['fielders']:
                                if 'name' in fielder:
                                    if wicket.get('kind') == 'run out':
                                        player_stats[fielder['name']]['run_outs'] += 1
                                    elif wicket.get('kind') == 'stumped':
                                        player_stats[fielder['name']]['stumpings'] += 1
                                    else: # Assumes other dismissals involving fielders are catches
                                        player_stats[fielder['name']]['catches'] += 1
    
    records = []
    for player, stats in player_stats.items():
        stats = player_stats[player]
        record = {
            'match_id': match_id,
            'player': player,
            'team': player_to_team.get(player, 'Unknown'),
            'date': info.get('dates', [None])[0],
            'venue': info.get('venue'),
            'city': info.get('city'),
            **stats
        }
        records.append(record)
    return records

def process_all_matches(raw_data_dir, interim_data_path):
    raw_path = Path(raw_data_dir)
    interim_path = Path(interim_data_path)
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    json_files = list(raw_path.glob('*.json'))
    
    logging.info(f"Found {len(json_files)} JSON files to process.")
    all_match_data = []
    for file in tqdm(json_files, desc="Processing Cricsheet JSON files"):
        try:
            match_records = parse_match_json(file)
            if match_records:
                all_match_data.extend(match_records)
        except Exception as e:
            logging.error(f"Failed to process file {file.name}: {e}")

    df = pd.DataFrame(all_match_data)
    stat_cols = ['runs_scored', 'balls_faced', 'fours', 'sixes', 'balls_bowled', 'runs_conceded', 'wickets', 'catches', 'stumpings', 'run_outs']
    for col in df.columns:
        if col in stat_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        elif df[col].dtype == 'object' and col not in ['match_id', 'player', 'date', 'venue', 'city']:
            df[col] = df[col].fillna('')

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    logging.info("Saving processed data to interim Parquet file...")
    df.to_parquet(interim_path, index=False)
    logging.info(f"Successfully saved enhanced data to {interim_path}")

if __name__ == '__main__':
    RAW_DATA_DIR = 'data/raw/cricsheet_data'
    INTERIM_DATA_PATH = 'data/interim/player_match_stats.parquet'
    process_all_matches(RAW_DATA_DIR, INTERIM_DATA_PATH)