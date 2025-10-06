import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def infer_player_roles(raw_data_dir):
    """
    Infer player roles from T20 MATCHES ONLY.
    """
    raw_path = Path(raw_data_dir)
    json_files = list(raw_path.glob('*.json'))
    
    if not json_files:
        logging.error(f"No JSON files found in {raw_data_dir}.")
        return pd.DataFrame(columns=['player', 'role'])
    
    logging.info(f"Analyzing {len(json_files)} files for T20 matches...")
    
    player_stats = {}
    t20_matches = 0
    skipped_matches = 0
    
    for file in tqdm(json_files, desc="Inferring Player Roles"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Could not decode JSON from {file.name}")
            continue
        
        info = data.get('info', {})
        
        # CRITICAL: Filter for T20 only
        match_type = info.get('match_type', '').lower()
        match_type_number = info.get('match_type_number', None)
        
        is_t20 = (match_type_number == 3 or 
                  't20' in match_type or 
                  'twenty20' in match_type.replace(' ', ''))
        
        if not is_t20:
            skipped_matches += 1
            continue
        
        t20_matches += 1
        
        # Initialize players
        if 'players' in info:
            for team_players in info['players'].values():
                for player in team_players:
                    if player not in player_stats:
                        player_stats[player] = {
                            'matches': 0,
                            'runs': 0,
                            'balls_faced': 0,
                            'wickets': 0,
                            'balls_bowled': 0,
                            'stumpings': 0
                        }
                    player_stats[player]['matches'] += 1
        
        # Collect stats
        for inning in data.get('innings', []):
            for over in inning.get('overs', []):
                for delivery in over.get('deliveries', []):
                    batter = delivery.get('batter')
                    bowler = delivery.get('bowler')
                    
                    if batter in player_stats:
                        player_stats[batter]['runs'] += delivery['runs'].get('batter', 0)
                        player_stats[batter]['balls_faced'] += 1
                    
                    if bowler in player_stats:
                        player_stats[bowler]['balls_bowled'] += 1
                        
                        if 'wickets' in delivery:
                            for wicket in delivery['wickets']:
                                kind = wicket.get('kind', '')
                                
                                if kind not in ['run out', 'retired hurt', 'retired out']:
                                    player_stats[bowler]['wickets'] += 1
                                
                                if kind == 'stumped':
                                    for fielder_info in wicket.get('fielders', []):
                                        keeper = fielder_info.get('name')
                                        if keeper and keeper in player_stats:
                                            player_stats[keeper]['stumpings'] += 1
    
    logging.info(f"Processed {t20_matches} T20 matches, skipped {skipped_matches} non-T20 matches")
    
    if not player_stats:
        logging.error("No player statistics collected from T20 matches!")
        return pd.DataFrame(columns=['player', 'role'])
    
    # Classify players
    player_roles = []
    
    for player, stats in player_stats.items():
        if stats['matches'] == 0:
            continue
        
        runs_per_match = stats['runs'] / stats['matches']
        balls_faced_per_match = stats['balls_faced'] / stats['matches']
        wickets_per_match = stats['wickets'] / stats['matches']
        overs_per_match = (stats['balls_bowled'] / 6) / stats['matches']
        
        # T20-specific thresholds
        is_wk = stats['stumpings'] > 0
        is_regular_bat = balls_faced_per_match >= 8  # Bats in top/middle order
        is_regular_bowl = overs_per_match >= 2  # Bowls 2+ overs per match
        is_pure_bat = is_regular_bat and overs_per_match < 1
        is_pure_bowl = is_regular_bowl and balls_faced_per_match < 5
        
        if is_wk:
            role = 'WK'
        elif is_regular_bat and is_regular_bowl:
            role = 'AR'
        elif is_pure_bowl:
            role = 'BOWL'
        elif is_pure_bat or is_regular_bat:
            role = 'BAT'
        elif overs_per_match >= 2:
            role = 'BOWL'
        else:
            role = 'AR'
        
        player_roles.append({
            'player': player,
            'role': role,
            'matches': stats['matches'],
            'runs_per_match': round(runs_per_match, 1),
            'balls_faced_per_match': round(balls_faced_per_match, 1),
            'wickets_per_match': round(wickets_per_match, 2),
            'overs_per_match': round(overs_per_match, 1)
        })
    
    roles_df = pd.DataFrame(player_roles)
    
    logging.info(f"\nRole inference complete for {len(roles_df)} players from T20 matches")
    logging.info(f"\nRole distribution:")
    for role, count in roles_df['role'].value_counts().items():
        logging.info(f"  {role}: {count} players")
    
    return roles_df


if __name__ == '__main__':
    RAW_DATA_DIR = 'data/raw/cricsheet_data'
    OUTPUT_PATH = 'data/processed/player_roles.csv'
    
    roles_df = infer_player_roles(RAW_DATA_DIR)
    
    if not roles_df.empty:
        roles_df.drop_duplicates(subset=['player'], inplace=True)
        
        roles_df[['player', 'role']].to_csv(OUTPUT_PATH, index=False)
        
        detailed_output = OUTPUT_PATH.replace('.csv', '_detailed.csv')
        roles_df.to_csv(detailed_output, index=False)
        
        logging.info(f"\nSuccessfully saved T20 player roles to {OUTPUT_PATH}")
        logging.info(f"Detailed stats saved to {detailed_output}")
