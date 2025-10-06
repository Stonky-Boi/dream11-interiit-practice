import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def infer_player_roles(raw_data_dir):
    """
    Infers player roles with more granular logic by analyzing their actions
    across all match files.
    """
    raw_path = Path(raw_data_dir)
    json_files = list(raw_path.glob('*.json'))

    if not json_files:
        logging.error(f"No JSON files found in {raw_data_dir}.")
        return pd.DataFrame(columns=['player', 'role'])

    batters = set()
    bowlers = set()
    keepers = set()

    logging.info(f"Analyzing {len(json_files)} match files to infer roles...")

    for file in tqdm(json_files, desc="Inferring Player Roles"):
        with open(file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Could not decode JSON from {file.name}")
                continue
        
        for inning in data.get('innings', []):
            for over in inning.get('overs', []):
                for delivery in over.get('deliveries', []):
                    # Add players to sets based on their actions
                    batters.add(delivery['batter'])
                    batters.add(delivery['non_striker'])
                    bowlers.add(delivery['bowler'])

                    # Identify keepers from stumping actions
                    if 'wickets' in delivery:
                        for wicket in delivery['wickets']:
                            if wicket.get('kind') == 'stumped' and 'fielders' in wicket:
                                for fielder in wicket['fielders']:
                                    if 'name' in fielder:
                                        keepers.add(fielder['name'])
    
    # Get a complete list of all players involved
    all_players = sorted(list(batters | bowlers))
    
    player_roles = []
    for player in all_players:
        is_batter = player in batters
        is_bowler = player in bowlers
        
        if player in keepers:
            role = 'WK'
        elif is_batter and is_bowler:
            role = 'AR'
        elif is_bowler:
            role = 'BOWL'
        elif is_batter:
            role = 'BAT'
        else:
            role = 'Unknown' # Edge case for players who might appear but not bat/bowl
            
        player_roles.append({'player': player, 'role': role})

    logging.info(f"Role inference complete. Total players: {len(all_players)}")
    return pd.DataFrame(player_roles)

if __name__ == '__main__':
    RAW_DATA_DIR = 'data/raw/cricsheet_data'
    OUTPUT_PATH = 'data/processed/player_roles.csv'
    
    roles_df = infer_player_roles(RAW_DATA_DIR)
    
    if not roles_df.empty:
        roles_df = roles_df[roles_df['role'] != 'Unknown'] # Remove any unknown players
        roles_df.drop_duplicates(subset=['player'], inplace=True)
        roles_df.to_csv(OUTPUT_PATH, index=False)
        logging.info(f"Successfully generated and saved player roles to {OUTPUT_PATH}")