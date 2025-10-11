from __future__ import print_function
from pathlib import Path
from tqdm import tqdm
import json
import os
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print_length = 100

# ==================== CRICKETSTATS LIBRARY FUNCTIONS ====================
# From data_download.py (PRIORITY)

try:
    import cricketstats as cks
except ImportError:
    print("⚠️  Installing cricketstats library...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'cricketstats'])
    import cricketstats as cks

def download_cricsheet_database(data_dir='data'):
    """Download complete Cricsheet database"""
    import requests
    
    raw_dir = Path(data_dir) / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    database_path = raw_dir / 'all_json.zip'
    
    if database_path.exists():
        print(f"✓ Database already exists: {database_path}")
        file_size = database_path.stat().st_size / (1024 * 1024)
        print(f"  Size: {file_size:.1f} MB")
        return str(database_path)
    
    print("=" * print_length)
    print("DOWNLOADING CRICSHEET DATABASE")
    print("=" * print_length)
    print("⚠️  This may take 5-10 minutes depending on your connection")
    print("📦 Downloading all_json.zip from cricsheet.org...")
    
    url = 'https://cricsheet.org/downloads/all_json.zip'
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(database_path, 'wb') as f, tqdm(
            desc="Download Progress",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            ncols=100
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        print(f"✓ Downloaded successfully to: {database_path}")
        return str(database_path)
        
    except Exception as e:
        print(f"✗ Download failed: {str(e)}")
        print("\n📥 Please download manually:")
        print("   1. Visit: https://cricsheet.org/downloads/all_json.zip")
        print(f"   2. Save to: {database_path}")
        return None

def data_extracter(database, matchtype, from_date, to_date, data_dir='data'):
    """
    Extract match-level data from Cricsheet using cricketstats
    
    Args:
        database: Path to all_json.zip
        matchtype: List of match types ['ODI', 'ODM', 'T20', 'IT20']
        from_date: Tuple (year, month, day)
        to_date: Tuple (year, month, day)
        data_dir: Base data directory
    """
    search = cks.cricketstats.search(allplayers=True)
    
    # Optional Arguments
    betweenovers = []
    innings = []
    fielders = []
    oppositionbatters = []
    oppositionbowlers = []
    superover = None
    battingposition = []
    bowlingposition = []
    sex = []
    playerteams = []
    teammates = []
    oppositionteams = []
    venue = []
    teamtype = []
    event = []
    matchresult = None
    sumstats = False
    
    # Execute search
    search.stats(
        database, from_date, to_date, matchtype,
        betweenovers=betweenovers,
        innings=innings,
        sex=sex,
        playersteams=playerteams,
        oppositionbatters=oppositionbatters,
        oppositionbowlers=oppositionbowlers,
        oppositionteams=oppositionteams,
        venue=venue,
        event=event,
        matchresult=matchresult,
        superover=superover,
        battingposition=battingposition,
        bowlingposition=bowlingposition,
        fielders=fielders,
        sumstats=sumstats
    )
    
    # Get innings-level results
    data_var = "_".join(matchtype) + "_data"
    result_df = search.inningsresult
    
    # Save to CSV
    raw_dir = Path(data_dir) / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{data_var}.csv"
    
    result_df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(result_df):,} records to {output_path}")
    
    return result_df

def aggregate_points_json(database, matchtype, from_date, to_date, data_dir='data'):
    """
    Extract aggregate career statistics from Cricsheet
    
    Args:
        database: Path to all_json.zip
        matchtype: List of match types
        from_date: Tuple (year, month, day)
        to_date: Tuple (year, month, day)
        data_dir: Base data directory
    """
    search = cks.cricketstats.search(allplayers=True)
    
    # Optional Arguments (same as data_extracter)
    betweenovers = []
    innings = []
    fielders = []
    oppositionbatters = []
    oppositionbowlers = []
    superover = None
    battingposition = []
    bowlingposition = []
    sex = []
    playerteams = []
    teammates = []
    oppositionteams = []
    venue = []
    teamtype = []
    event = []
    matchresult = None
    sumstats = False
    
    # Execute search
    search.stats(
        database, from_date, to_date, matchtype,
        betweenovers=betweenovers,
        innings=innings,
        sex=sex,
        playersteams=playerteams,
        oppositionbatters=oppositionbatters,
        oppositionbowlers=oppositionbowlers,
        oppositionteams=oppositionteams,
        venue=venue,
        event=event,
        matchresult=matchresult,
        superover=superover,
        battingposition=battingposition,
        bowlingposition=bowlingposition,
        fielders=fielders,
        sumstats=sumstats
    )
    
    # Get aggregate results
    data_var = "_".join(matchtype) + "_data"
    result_df = search.result
    
    # Drop Maiden Overs column if exists
    if 'Maiden Overs' in result_df.columns:
        result_df = result_df.drop(columns=['Maiden Overs'])
    
    # Fill NaN with -1
    result_df.fillna(-1, inplace=True)
    
    # Convert to JSON format
    json_data = result_df.set_index('Players').to_dict(orient='index')
    
    # Save JSON
    processed_dir = Path(data_dir) / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    json_file_path = processed_dir / f"{data_var}_aggregate_data.json"
    
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    
    print(f"✓ Saved aggregate stats for {len(json_data):,} players to {json_file_path}")
    
    return json_data

# ==================== CUSTOM BALL-BY-BALL PARSER ====================
# From extract_data.py

def normalize_name(n):
    """Normalize player names"""
    return n.strip() if isinstance(n, str) else n

def load_json(path):
    """Load JSON file with encoding handling"""
    try:
        f = open(path, 'r', encoding='utf-8')
    except TypeError:
        f = open(path, 'r')
    with f:
        return json.load(f)

def collect_all_players(data):
    """Collect every player seen in squads and deliveries/fielders"""
    all_players = set()
    info = data.get('info', {})
    
    # From squad lists
    for team, plist in info.get('players', {}).items():
        for p in plist:
            nm = normalize_name(p)
            if nm:
                all_players.add(nm)
    
    # From deliveries
    innings = data.get('innings', [])
    has_structured_overs = len(innings) > 0 and isinstance(innings[0].get('overs', []), list)
    
    def add_names_from_delivery(d):
        for key in ('batter', 'bowler', 'non_striker'):
            if key in d:
                nm = normalize_name(d.get(key, ''))
                if nm:
                    all_players.add(nm)
        
        if 'wickets' in d or 'wicket' in d:
            wickets_data = d.get('wickets', [d.get('wicket')] if 'wicket' in d else [])
            for w in wickets_data:
                if w:
                    for f in w.get('fielders', []):
                        nm = normalize_name(f.get('name', ''))
                        if nm:
                            all_players.add(nm)
    
    if has_structured_overs:
        for inn in innings:
            for over in inn.get('overs', []):
                for d in over.get('deliveries', []):
                    add_names_from_delivery(d)
    else:
        for inn in innings:
            for ob in inn.get('deliveries', []):
                for _, d in ob.items():
                    add_names_from_delivery(d)
    
    all_players.discard('')
    return all_players

def parse_match(file_path):
    """Parse a single match JSON file for ball-by-ball stats"""
    data = load_json(file_path)
    match_id = os.path.basename(file_path).split('.')[0]
    
    info = data.get('info', {})
    dates = info.get('dates', [])
    date = datetime.strptime(dates[0], '%Y-%m-%d') if dates and isinstance(dates[0], str) else datetime.min
    
    balls_per_over = info.get('balls_per_over', 6)
    registry_people = info.get('registry', {}).get('people', {}) or {}
    name_to_pid = {normalize_name(n): pid for n, pid in registry_people.items()}
    
    all_players = collect_all_players(data)
    
    # Initialize stats
    batting = {p: {'runs':0,'sixes':0,'fours':0,'balls_faced':0,'dots':0} for p in all_players}
    bowling = {p: {'wickets':0,'runs_conceded':0,'balls_bowled':0} for p in all_players}
    fielding = {p: {'catches':0,'stumpings':0,'runouts':0} for p in all_players}
    
    def legal_ball(extras):
        return (extras.get('wides', 0) == 0) and (extras.get('noballs', 0) == 0)
    
    def credit_fielding_and_wickets(d):
        wickets_data = d.get('wickets', [d.get('wicket')] if 'wicket' in d else [])
        for w in wickets_data:
            if not w:
                continue
            
            kind = w.get('kind', '')
            bowler = normalize_name(d.get('bowler', ''))
            fielders_list = [normalize_name(f.get('name', '')) for f in w.get('fielders', []) if f.get('name')]
            
            if kind in ['bowled', 'lbw', 'hit wicket', 'caught', 'caught and bowled', 'stumped']:
                if bowler and bowler in bowling:
                    bowling[bowler]['wickets'] += 1
            
            if kind == 'caught':
                if fielders_list:
                    catcher = fielders_list[0]
                    if catcher in fielding:
                        fielding[catcher]['catches'] += 1
            elif kind == 'caught and bowled':
                if bowler and bowler in fielding:
                    fielding[bowler]['catches'] += 1
            elif kind == 'stumped':
                if fielders_list:
                    keeper = fielders_list[0]
                    if keeper in fielding:
                        fielding[keeper]['stumpings'] += 1
            elif kind == 'run out':
                if fielders_list:
                    f_runout = fielders_list[-1]
                    if f_runout in fielding:
                        fielding[f_runout]['runouts'] += 1
    
    def process_delivery(d):
        batter = normalize_name(d.get('batter', ''))
        bowler = normalize_name(d.get('bowler', ''))
        runs = d.get('runs', {}) or {}
        extras = d.get('extras', {}) or {}
        
        if batter and batter in batting:
            br = runs.get('batter', 0)
            bstats = batting[batter]
            bstats['runs'] += br
            if br == 4: bstats['fours'] += 1
            if br == 6: bstats['sixes'] += 1
            if legal_ball(extras):
                bstats['balls_faced'] += 1
                if (br == 0) and (extras.get('byes', 0) == 0) and (extras.get('legbyes', 0) == 0):
                    bstats['dots'] += 1
        
        if bowler and bowler in bowling:
            conceded = runs.get('batter', 0) + extras.get('wides', 0) + extras.get('noballs', 0)
            bowling[bowler]['runs_conceded'] += conceded
            if legal_ball(extras):
                bowling[bowler]['balls_bowled'] += 1
        
        if 'wickets' in d or 'wicket' in d:
            credit_fielding_and_wickets(d)
    
    # Process all deliveries
    innings = data.get('innings', [])
    has_structured_overs = len(innings) > 0 and isinstance(innings[0].get('overs', []), list)
    
    if has_structured_overs:
        for inn in innings:
            for over in inn.get('overs', []):
                for d in over.get('deliveries', []):
                    process_delivery(d)
    else:
        for inn in innings:
            for ob in inn.get('deliveries', []):
                for _, d in ob.items():
                    process_delivery(d)
    
    # Compile match stats
    match_stats = {}
    for name in all_players:
        pid = name_to_pid.get(name, name)
        bf = batting[name]['balls_faced']
        sr = (batting[name]['runs'] * 100.0 / float(bf)) if bf > 0 else 0.0
        bb = bowling[name]['balls_bowled']
        eco = (bowling[name]['runs_conceded'] / (float(bb) / float(balls_per_over))) if bb > 0 else 0.0
        
        match_stats[pid] = {
            'runs': batting[name]['runs'],
            'sixes': batting[name]['sixes'],
            'fours': batting[name]['fours'],
            'strike_rate': sr,
            'dots': batting[name]['dots'],
            'wickets': bowling[name]['wickets'],
            'economy': eco,
            'catches': fielding[name]['catches'],
            'stumpings': fielding[name]['stumpings'],
            'runouts': fielding[name]['runouts'],
        }
    
    return match_id, date, match_stats

def build_wide_format(all_matches, periods=('A3', 'A5', 'A8', 'AC', 'A0')):
    """Build wide format with historical averages"""
    period_map = {'A3': 3, 'A5': 5, 'A8': 8, 'AC': 'career', 'A0': 0}
    stat_names = ['runs','sixes','fours','strike_rate','dots','wickets','economy','catches','stumpings','runouts']
    
    player_history = {}
    for match_id, date, stats in all_matches:
        for pid, s in stats.items():
            player_history.setdefault(pid, []).append((match_id, s))
    
    rows = []
    for match_id, date, stats in all_matches:
        match_date_str = date.strftime("%Y-%m-%d") if date != datetime.min else ""
        
        for pid, current_stats in stats.items():
            hist = player_history.get(pid, [])
            idx = next((i for i, (mid, _) in enumerate(hist) if mid == match_id), None)
            prev_list = hist[:idx] if idx is not None else []
            
            row = {'match_id': match_id, 'match_date': match_date_str, 'player_id': pid}
            
            for period_name in periods:
                n = period_map[period_name]
                
                if n == 0:
                    period_stats = current_stats
                elif n == 'career':
                    if not prev_list:
                        period_stats = {stat: 0.0 for stat in stat_names}
                    else:
                        denom = float(len(prev_list))
                        period_stats = {stat: sum(s[stat] for _, s in prev_list)/denom for stat in stat_names}
                else:
                    last_n = prev_list[-n:] if len(prev_list) >= n else prev_list
                    if not last_n:
                        period_stats = {stat: 0.0 for stat in stat_names}
                    else:
                        denom = float(len(last_n))
                        period_stats = {stat: sum(s[stat] for _, s in last_n)/denom for stat in stat_names}
                
                for stat in stat_names:
                    row[f'{period_name}_{stat}'] = period_stats[stat]
            
            rows.append(row)
    
    return rows

def extract_ball_by_ball(directory, output_prefix='ball_by_ball', data_dir='data'):
    """Extract ball-by-ball data from JSON directory"""
    print("\n" + "=" * print_length)
    print("BALL-BY-BALL EXTRACTION (CUSTOM PARSER)")
    print("=" * print_length)
    
    all_matches = []
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    print(f"Processing {len(json_files)} JSON files...")
    
    for filename in tqdm(json_files, desc="Parsing matches", ncols=100):
        path = os.path.join(directory, filename)
        try:
            match_id, date, stats = parse_match(path)
            all_matches.append((match_id, date, stats))
        except Exception as e:
            print(f"\n⚠️  Error processing {filename}: {e}")
    
    # Sort by date
    all_matches.sort(key=lambda x: (x[1], x[0]))
    
    # Build wide format
    print("\nBuilding wide format with period averages...")
    rows = build_wide_format(all_matches, periods=('A3', 'A5', 'A8', 'AC', 'A0'))
    df = pd.DataFrame(rows)
    
    # Convert match_date to datetime
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    df = df.sort_values(['match_date', 'player_id'], ascending=[True, True])
    
    # Round float columns
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].round(0).astype(int)
    
    # Save full dataset
    interim_dir = Path(data_dir) / 'interim'
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    full_path = interim_dir / f"{output_prefix}_full.csv"
    df.to_csv(full_path, index=False)
    
    print(f"✓ Ball-by-ball data saved: {len(df)} rows, {len(df.columns)} columns")
    print(f"✓ Saved to: {full_path}")
    
    return df

# ==================== MAIN PIPELINE ====================

def run_extraction_pipeline(data_dir='data', extract_ball_by_ball_data=False):
    """Run complete data extraction pipeline - ALL DATA"""
    print("\n" + "=" * print_length)
    print("DREAM11 INTER-IIT DATA EXTRACTION PIPELINE")
    print("=" * print_length)
    print("Using cricketstats library + custom ball-by-ball parser")
    print("Extracting ALL available data (no date filtering)")
    print("Training cutoff will be enforced in train_model.py")
    print("=" * print_length)
    
    # Step 1: Download database
    print("\n[STEP 1/6] Downloading Cricsheet Database")
    print("-" * print_length)
    database = download_cricsheet_database(data_dir)
    
    if database is None:
        print("⚠️  Cannot proceed without database. Please download manually.")
        return
    
    # Step 2: Define date ranges
    from_date = (2010, 1, 1)  # Start from 2010
    to_date = (2030, 12, 31)  # Far future = all data
    
    print(f"\n✓ Extracting data from {from_date} to {to_date}")
    print("✓ Effectively extracting ALL available data")
    
    # Step 3: Extract ODI data
    print("\n[STEP 2/6] Extracting ODI Match Data (ALL)")
    print("-" * print_length)
    matchtype_odi = ["ODI", "ODM"]
    odi_data = data_extracter(database, matchtype_odi, from_date, to_date, data_dir)
    
    # Step 4: Extract T20 data
    print("\n[STEP 3/6] Extracting T20 Match Data (ALL)")
    print("-" * print_length)
    matchtype_t20 = ["T20"]
    t20_data = data_extracter(database, matchtype_t20, from_date, to_date, data_dir)
    
    # Step 5: Extract aggregate ODI data
    print("\n[STEP 4/6] Extracting ODI Aggregate Statistics (ALL)")
    print("-" * print_length)
    odi_aggregate = aggregate_points_json(database, matchtype_odi, from_date, to_date, data_dir)
    
    # Step 6: Extract aggregate T20 data
    print("\n[STEP 5/6] Extracting T20 Aggregate Statistics (ALL)")
    print("-" * print_length)
    t20_aggregate = aggregate_points_json(database, matchtype_t20, from_date, to_date, data_dir)
    
    # Step 7: Optional ball-by-ball extraction
    if extract_ball_by_ball_data:
        print("\n[STEP 6/6] Extracting Ball-by-Ball Data (Custom Parser)")
        print("-" * print_length)
        # Need to extract JSON files from zip first
        import zipfile
        json_dir = Path(data_dir) / 'raw' / 'cricsheet_data'
        json_dir.mkdir(parents=True, exist_ok=True)
        
        zip_path = Path(data_dir) / 'raw' / 'all_json.zip'
        if zip_path.exists():
            print("Extracting JSON files from all_json.zip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(json_dir)
            
            # Find the actual JSON directory
            json_subdir = None
            for item in json_dir.iterdir():
                if item.is_dir():
                    json_subdir = item
                    break
            
            if json_subdir:
                extract_ball_by_ball(str(json_subdir), data_dir=data_dir)
            else:
                print("⚠️  Could not find JSON subdirectory")
        else:
            print("⚠️  all_json.zip not found, skipping ball-by-ball extraction")
    
    # Summary
    print("\n" + "=" * print_length)
    print("EXTRACTION SUMMARY")
    print("=" * print_length)
    print(f"✓ ODI match data: {len(odi_data):,} player-innings records")
    print(f"✓ T20 match data: {len(t20_data):,} player-innings records")
    print(f"✓ Total records: {len(odi_data) + len(t20_data):,}")
    print(f"✓ ODI aggregate: {len(odi_aggregate):,} players")
    print(f"✓ T20 aggregate: {len(t20_aggregate):,} players")
    print(f"✓ Data saved to: {data_dir}/")
    
    print("\n" + "=" * print_length)
    print("✓✓✓ DATA EXTRACTION COMPLETE ✓✓✓")
    print("=" * print_length)
    print("\nFiles created:")
    print(f"  • {data_dir}/raw/ODI_ODM_data.csv")
    print(f"  • {data_dir}/raw/T20_data.csv")
    print(f"  • {data_dir}/processed/ODI_ODM_data_aggregate_data.json")
    print(f"  • {data_dir}/processed/T20_data_aggregate_data.json")
    if extract_ball_by_ball_data:
        print(f"  • {data_dir}/interim/ball_by_ball_full.csv")
    
    print("\n⚠️  IMPORTANT:")
    print("  • ALL available data has been extracted")
    print("  • Training cutoff (2024-06-30) will be enforced in train_model.py")
    print("  • This allows proper train/val/test temporal splitting")
    
    print("\nNext step: python data_processing/feature_engineering.py")

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dream11 Data Extraction Pipeline')
    parser.add_argument('--data_dir', default='data', help='Base data directory')
    parser.add_argument('--ball_by_ball', action='store_true', help='Extract ball-by-ball data (slow)')
    
    args = parser.parse_args()
    
    run_extraction_pipeline(data_dir=args.data_dir, extract_ball_by_ball_data=args.ball_by_ball)

if __name__ == '__main__':
    main()