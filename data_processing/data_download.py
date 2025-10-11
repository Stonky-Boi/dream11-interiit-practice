"""
Data Download Script for Dream11 Inter-IIT Project
Uses cricketstats library following Silver Medal Team approach
Closely follows Cricsheet JSON format
"""

from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print_length = 100

try:
    import cricketstats as cks
except ImportError:
    print("⚠️  Installing cricketstats library...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'cricketstats'])
    import cricketstats as cks

def download_cricsheet_database(data_dir='data'):
    """Download the complete Cricsheet all_json.zip database"""
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
    print("📦  Downloading all_json.zip from cricsheet.org...")
    
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
    Extract match-level innings data using cricketstats library
    Follows Silver Medal Team's exact approach
    
    Args:
        database: Path to all_json.zip
        matchtype: List of match types ['ODI', 'ODM', 'T20', 'IT20', 'Test', 'MDM']
        from_date: Tuple (year, month, day)
        to_date: Tuple (year, month, day)
        data_dir: Base data directory
    """
    search = cks.cricketstats.search(allplayers=True)
    
    # Optional Arguments (as per Silver Medal Team)
    betweenovers = []  # Search only these overs
    innings = []  # Search these innings. Options: 1, 2, 3, 4
    fielders = []  # Search bowling stats involves these fielders
    oppositionbatters = []  # Search overs where players have bowled against certain batters
    oppositionbowlers = []  # Search overs where players have batted against certain bowlers
    superover = None  # Search normal innings or superover innings
    battingposition = []  # Search stats at certain position in batting order
    bowlingposition = []  # Search stats at certain position in bowling order
    
    # Match related arguments
    sex = []  # Search only matches of certain sex. Options: "male", "female"
    playerteams = []  # Search matches where players have played in certain teams
    teammates = []  # Search matches where certain teammates play
    oppositionteams = []  # Search matches where opposition is only certain teams
    venue = []  # Search matches played only at these venues
    teamtype = []  # Search only for particular type of teams
    event = []  # Search matches played as part of these Leagues or Tournaments
    matchresult = None  # Search matches where players or teams have these results
    sumstats = False  # When True, adds an "all players" row at end
    
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
    Extract aggregate player statistics and save as JSON
    Follows Silver Medal Team's exact approach
    
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
    
    # Drop Maiden Overs column if exists (as per Silver Medal Team)
    if 'Maiden Overs' in result_df.columns:
        result_df = result_df.drop(columns=['Maiden Overs'])
    
    # Fill NaN with -1 (as per Silver Medal Team)
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

def run_extraction_pipeline(cutoff_date='2024-06-30', data_dir='data'):
    """
    Run complete extraction pipeline
    Extracts both training and test data (if available)
    """
    print("\n" + "=" * print_length)
    print("DREAM11 INTER-IIT DATA EXTRACTION PIPELINE")
    print("=" * print_length)
    print("Using cricketstats library (Silver Medal Team approach)")
    print(f"Training cutoff: {cutoff_date}")
    print("=" * print_length)
    
    # Step 1: Download database
    print("\n[STEP 1/5] Downloading Cricsheet Database")
    print("-" * print_length)
    database = download_cricsheet_database(data_dir)
    
    if database is None:
        print("⚠️  Cannot proceed without database. Please download manually.")
        return
    
    # Step 2: Define date ranges
    from_date = (2010, 1, 1)
    cutoff = datetime.strptime(cutoff_date, '%Y-%m-%d')
    to_date_train = (cutoff.year, cutoff.month, cutoff.day)
    
    print(f"\n✓ Training data: {from_date} to {to_date_train}")
    
    # Step 3: Extract ODI data
    print("\n[STEP 2/5] Extracting ODI Match Data (Training)")
    print("-" * print_length)
    matchtype_odi = ["ODI", "ODM"]
    odi_data = data_extracter(database, matchtype_odi, from_date, to_date_train, data_dir)
    
    # Step 4: Extract T20 data
    print("\n[STEP 3/5] Extracting T20 Match Data (Training)")
    print("-" * print_length)
    matchtype_t20 = ["T20"]
    t20_data = data_extracter(database, matchtype_t20, from_date, to_date_train, data_dir)
    
    # Step 5: Extract aggregate ODI data
    print("\n[STEP 4/5] Extracting ODI Aggregate Statistics")
    print("-" * print_length)
    odi_aggregate = aggregate_points_json(database, matchtype_odi, from_date, to_date_train, data_dir)
    
    # Step 6: Extract aggregate T20 data
    print("\n[STEP 5/5] Extracting T20 Aggregate Statistics")
    print("-" * print_length)
    t20_aggregate = aggregate_points_json(database, matchtype_t20, from_date, to_date_train, data_dir)
    
    # Summary
    print("\n" + "=" * print_length)
    print("EXTRACTION SUMMARY")
    print("=" * print_length)
    print(f"✓ ODI match data: {len(odi_data):,} player-innings records")
    print(f"✓ T20 match data: {len(t20_data):,} player-innings records")
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
    print("\nNext step: python data_processing/feature_engineering.py")

def main():
    """Main execution"""
    run_extraction_pipeline(cutoff_date='2024-06-30', data_dir='data')

if __name__ == '__main__':
    main()